#%%
import torch
from Fast_MRI_dataloader import create_dataloaders
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#%%
# define parameters
data_loc = os.path.dirname(os.path.realpath(__file__)) #change the datalocation to something that works for you
data_loc = os.path.join(data_loc,'Fast_MRI_Knee')
batch_size = 2

# create dataloaders
train_loader, test_loader = create_dataloaders(data_loc, batch_size)

# go over the dataset
gt = []
for i,(kspace, M, gt) in enumerate(tqdm(test_loader)):
    continue


#%%
# Create function to calculate the K-space of MRI images
def image_to_kspace(image):
    kspace = torch.fft.fftshift(torch.fft.fft2(image, norm="forward"))
    return kspace

#%%
# Create function to calculate partial kspace of MRI images
def image_to_partialkspace(kspace, mask):
    partialkspace = mask * kspace
    return partialkspace
#%%
# Create function for accelerating measurement MRI
def pkspace_to_image(pkspace):
    image = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(pkspace), norm="forward"))
    return image

#%% 
# Get the kspace, the pkspace, the max and min of kspace_plot_friendly 
kspce = image_to_kspace(gt)
pkspce = image_to_partialkspace(kspce, M)
acc_MRI = pkspace_to_image(pkspce)

pkspace_plot_friendly = torch.log(torch.abs(pkspce)+1e-20)
kspace_plot_friendly = torch.log(torch.abs(kspce)+1e-20)

vmin = torch.min(kspace_plot_friendly)
vmax = torch.max(kspace_plot_friendly)

#%%
# Plot the ground truth, full k-space, sample mask, partial k-space and accelerated MRI
plt.figure(figsize = (10,10))
plt.subplot(1,5,1)
plt.imshow(gt[0,:,:],cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Ground truth MRI')

plt.subplot(1,5,2)
plt.imshow(kspace_plot_friendly[0,:,:],vmin=vmin, vmax=vmax, interpolation='nearest')
plt.xticks([])
plt.yticks([])
plt.title('Full K-space')

plt.subplot(1,5,3)
plt.imshow(M[0,:,:], interpolation='nearest')
plt.xticks([])
plt.yticks([])
plt.title('Sampling mask')

plt.subplot(1,5,4)
plt.imshow(pkspace_plot_friendly[0,:,:],vmin=vmin, vmax=vmax, interpolation='nearest')
plt.xticks([])
plt.yticks([])
plt.title('Partial K-space')

plt.subplot(1,5,5)
plt.imshow(acc_MRI[0,:,:],cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Accelerated MRI')
plt.savefig('Knee_MRI_Acceleration.png')
plt.show()


# %% ISTA
def softthreshold(x,shrinkage):
    # Initialize output
    x_threshold = x.clone()

    # Thresholding
    x_threshold[x > shrinkage] = ((torch.abs(x[x > shrinkage]) - shrinkage)/torch.abs(x[x > shrinkage]))*x[x > shrinkage]
    x_threshold[x <= shrinkage] = 0

    return x_threshold

def ISTA(mu,shrinkage,K,kspace,M):
    # Identity matrix A
    A = M.clone()
    # Identity matrix I
    I = torch.eye(kspace.shape[2])
    for i in tqdm(range(K)):
        if i == 0:
            input = pkspace_to_image(kspace)
        else:
            input = x_new
        x_old = input
        x_new = softthreshold((mu* pkspace_to_image(kspace) + torch.matmul(I-mu*A*torch.transpose(A,0,1),x_old)),shrinkage)
    return x_new

# parameters
mu = 0.09
shrinkage = 0.0001
K = 10
# reconstructed image
recon_image = ISTA(mu,shrinkage,K,kspace,M)

plt.figure(figsize = (10,10))
plt.subplot(1,2,1)
plt.imshow(gt[0,:,:],cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Ground truth MRI')

plt.subplot(1,2,2)
plt.imshow(recon_image[0,:,:],cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Reconstructed MRI')
plt.show()