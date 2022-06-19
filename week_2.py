import torch
from Fast_MRI_dataloader import create_dataloaders
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import tqdm
import matplotlib.pyplot as plt

# define parameters
data_loc = os.path.dirname(os.path.realpath(__file__)) #change the datalocation to something that works for you
data_loc = os.path.join(data_loc,'Fast_MRI_Knee')
batch_size = 2

# create dataloaders
train_loader, test_loader = create_dataloaders(data_loc, batch_size)

# go over the dataset
gt = []
for i,(kspace, M, gt) in enumerate(test_loader):
    continue



# Create function to calculate the K-space of MRI images
def image_to_kspace(image):
    kspace = torch.fft.fftshift(torch.fft.fft2(image, norm="forward"))
    return kspace

# Create function to calculate partial kspace of MRI images
def image_to_partialkspace(kspace, mask):
    partialkspace = mask * kspace
    return partialkspace


kspce = image_to_kspace(gt)
pkspce = image_to_partialkspace(kspce, M)

pkspace_plot_friendly = torch.log(torch.abs(pkspce)+1e-20)
kspace_plot_friendly = torch.log(torch.abs(kspce)+1e-20)

vmin = torch.min(kspace_plot_friendly)
vmax = torch.max(kspace_plot_friendly)

plt.figure(figsize = (10,10))
plt.subplot(1,4,1)
plt.imshow(gt[0,:,:],cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('ground truth')

plt.subplot(1,4,2)
plt.imshow(kspace_plot_friendly[0,:,:],vmin=vmin, vmax=vmax, interpolation='nearest')
plt.xticks([])
plt.yticks([])
plt.title('k-space')

plt.subplot(1,4,3)
plt.imshow(M[0,:,:], interpolation='nearest')
plt.xticks([])
plt.yticks([])
plt.title('measurement mask')

plt.subplot(1,4,4)
plt.imshow(pkspace_plot_friendly[0,:,:],vmin=vmin, vmax=vmax, interpolation='nearest')
plt.xticks([])
plt.yticks([])
plt.title('partial k-space')
plt.show()