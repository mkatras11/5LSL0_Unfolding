# %% imports
# libraries
from re import A
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from sklearn import metrics
from torch.nn import functional as F
# local imports
import MNIST_dataloader

# %% set torches random seed
torch.random.manual_seed(0)

# %% preperations
# define parameters
data_loc = os.path.dirname(os.path.realpath(__file__))
batch_size = 64

mu = 0.09
shrinkage = 0.01
K = 1

# get dataloader
train_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

# %% HINT
#hint: if you do not care about going over the data in mini-batches but rather want the entire dataset use:
x_clean_train = train_loader.dataset.Clean_Images
x_noisy_train = train_loader.dataset.Noisy_Images
labels_train  = train_loader.dataset.Labels

x_clean_test  = test_loader.dataset.Clean_Images
x_noisy_test  = test_loader.dataset.Noisy_Images
labels_test   = test_loader.dataset.Labels

# use these 10 examples as representations for all digits
x_clean_example = x_clean_test[0:10,:,:,:]
x_noisy_example = x_noisy_test[0:10,:,:,:]
labels_example = labels_test[0:10]

# %% ISTA
def softthreshold(x,shrinkage):
    # Initialize output
    x_threshold = x.clone()
    # Thresholding
    for k in range(x.shape[0]):
        for i in range(x.shape[2]):
            for j in range(x.shape[3]):
                if x[k,0,i,j] > shrinkage:
                    x_threshold[k,0,i,j] = ((torch.abs(x[k,0,i,j]) - shrinkage)/torch.abs(x[k,0,i,j]))*x[k,0,i,j]
                else:
                    x_threshold[k,0,i,j] = 0
    return x_threshold

def ISTA(mu,shrinkage,K,y):
    # Identity matrix A
    A = torch.eye(y.shape[2])
    # Identity matrix I
    I = torch.eye(y.shape[2])

    for i in tqdm(range(K)):
        if i == 0:
            input = y
        else:
            input = x_new
        x_old = input
        x_new = softthreshold((mu*torch.matmul(A,y) + torch.matmul(I-mu*A*torch.transpose(A,0,1),x_old)),shrinkage)
    return x_new


# %% Make the predictions
x_clean_pred = ISTA(mu,shrinkage,K,x_noisy_example)
# %% Show the 10 examples of the noisy images and the corresponding denoised images and the ground truth
# Plot the noisy and the predicted clean image
# show the examples in a plot
plt.figure(figsize=(12,3))
plt.subplot(3,10,6)
plt.title("Noisy Examples",loc='right')
for i in range(10):
    plt.subplot(3,10,i+1)
    plt.imshow(x_noisy_example[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.subplot(3,10,16)
plt.title("Predicted Examples",loc='right')
for i in range(10):
    plt.subplot(3,10,i+11)
    plt.imshow(x_clean_pred[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.subplot(3,10,26)
plt.title("Clean Examples",loc='right')
for i in range(10):
    plt.subplot(3,10,i+21)
    plt.imshow(x_clean_example[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])    
plt.tight_layout()
plt.show()

# %% Compute the MSE and the accuracy of the denoising
# Compute the MSE
mse_example = F.mse_loss(x_clean_example,x_clean_pred)
print('MSE example: ', mse_example)

# %% Entire dataset
# Run the model on the entire dataset over batches 
for batch_idx, (x_clean, x_noisy, labels) in enumerate(tqdm(test_loader)):
    x_clean_pred_test = ISTA(mu,shrinkage,K,x_noisy)

# %% MSE
# Compute the MSE for the training and test set
mse_test = F.mse_loss(x_clean_pred_test,x_clean)
print("MSE test: ", mse_test)


# %% plot results
# Plot the noisy and the predicted clean image
# show the examples in a plot
plt.figure(figsize=(12,3))
plt.subplot(3,10,6)
plt.title("First 10 noisy images",loc='right')
for i in range(10):
    plt.subplot(3,10,i+1)
    plt.imshow(x_noisy_test[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.subplot(3,10,16)
plt.title("First 10 predicted images",loc='right')
for i in range(10):
    plt.subplot(3,10,i+11)
    plt.imshow(x_clean_pred_test[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.subplot(3,10,26)
plt.title("First 10 clean images",loc='right')
for i in range(10):
    plt.subplot(3,10,i+21)
    plt.imshow(x_clean_test[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])    
plt.tight_layout()
plt.savefig("data_examples_results.png",dpi=300,bbox_inches='tight')
plt.show()


# %%
