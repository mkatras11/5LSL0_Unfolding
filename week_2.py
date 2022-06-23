#%%
from operator import index
import torch
from Fast_MRI_dataloader import create_dataloaders
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torch.fft import fft2, fftshift, ifft2, ifftshift

#%%
# define parameters
data_loc = os.path.dirname(os.path.realpath(__file__)) #change the datalocation to something that works for you
data_loc = os.path.join(data_loc,'Fast_MRI_Knee')
batch_size = 6

# create dataloaders
train_loader, test_loader = create_dataloaders(data_loc, batch_size)

# go over the dataset
gt = []
for i,(kspace, M, gt) in enumerate(tqdm(test_loader)):
    if i == 1:
        break
    continue


#%%
# Create function to calculate the K-space of MRI images
def image_to_kspace(image):
    kspace = fftshift(fft2(image, dim=(1, 2)), dim=(1, 2))
    return kspace

#%%
# Create function to calculate partial kspace of MRI images
def image_to_partialkspace(kspace, mask):
    partialkspace = torch.mul(kspace, mask)
    return partialkspace
#%%
# Create function for accelerating measurement MRI
def pkspace_to_image(pkspace):
    image = torch.abs(ifft2(ifftshift(pkspace, dim=(1, 2)), dim=(1, 2)))
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
    index = torch.abs(x) > shrinkage
    x[index] = x[index] * (torch.abs(x[index]) - shrinkage)/torch.abs(x[index])
    x[~index] = 0

    return x

def ISTA(mu,shrinkage,K,kspace,M):
    y = torch.abs(ifft2(kspace, dim=(1, 2)))
    input = y
    f_y = kspace

    for i in range(K):

        #softthreshold
        input = softthreshold(input, shrinkage)

        #data consistency 
        f_x = image_to_kspace(input)         
        pf_x = image_to_partialkspace(f_x, M)

        output = f_x - mu * pf_x + mu * f_y
        x = torch.abs(ifft2(output, dim=(1, 2)))

    return x

# parameters
mu = 0.2
shrinkage = 0.01
K = 30
# reconstructed image
recon_image = ISTA(mu,shrinkage,K,kspace,M)
recon_image_2 = pkspace_to_image(kspace)

plt.figure(figsize = (10,10))
plt.subplot(1,3,1)
plt.imshow(gt[0,:,:],cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Ground truth MRI')

plt.subplot(1,3,2)
plt.imshow(recon_image[0,:,:],cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Reconstructed MRI')

plt.subplot(1,3,3)
plt.imshow(recon_image_2[0,:,:],cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Noisy MRI')

plt.show()

def plot_ex4c(test_acc_mri, test_x_out, test_gt, save_path):

    plt.figure(figsize = (10, 6))
    for i in range(5):
        plt.subplot(3,5,i+1)
        plt.imshow(test_acc_mri[i+1,:,:], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 2:
            plt.title('Initial Reconstruction')

        plt.subplot(3,5,i+6)
        plt.imshow(test_x_out[i+1,:,:], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 2:
            plt.title('Reconstruction With ISTA')

        plt.subplot(3,5,i+11)
        plt.imshow(test_gt[i+1,:,:], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 2:
            plt.title('Ground Truth')

    plt.savefig(f"{save_path}", dpi=300, bbox_inches='tight')
    plt.show()


plot_ex4c(recon_image_2, recon_image, gt,'ex4c.png')


def calculate_loss_ista(data_loader, criterion, mu, shrinkage, K, apply_ista=True):
    """
    Calculate the loss on the given data set.
    -------
    data_loader : torch.utils.data.DataLoader
        Data loader to use for the data set.
    criterion : torch.nn.modules.loss
        Loss function to use.
    device : torch.device
        Device to use for the model.
    -------
    loss : float    
        The loss on the data set.
    """

    # initialize loss
    loss = 0

    # loop over batches
    for i, (partial_kspace, M, gt_mri) in enumerate(tqdm(data_loader, position=0, leave=False, ascii=False)):

            # forward pass
            if apply_ista:
                ista_mri = ISTA(mu, shrinkage, K, partial_kspace, M)
            else:
                ista_mri = torch.abs(ifft2(partial_kspace, dim=(1, 2)))
           
            # calculate loss
            loss += criterion(ista_mri, gt_mri)

    # return the loss
    return loss / len(data_loader)

mse = torch.nn.MSELoss()
mse_loss_accelerated_MRI = calculate_loss_ista(test_loader, mse, mu, shrinkage, K, apply_ista=False)
print(f"MSE loss on accelerated MRI images: {mse_loss_accelerated_MRI}")

# calculate mse on output of ISTA
mse_loss_ista = calculate_loss_ista(test_loader, mse, mu, shrinkage, K, apply_ista=True)
print(f"MSE loss ISTA out: {mse_loss_ista}") 