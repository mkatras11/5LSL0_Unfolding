# %% imports
# libraries
from re import A
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from sklearn import metrics
# local imports
import MNIST_dataloader

# %% set torches random seed
torch.random.manual_seed(0)

# %% preperations
# define parameters
data_loc = os.path.dirname(os.path.realpath(__file__))
batch_size = 64

mu = 1e-3
shrinkage = 0.0001
K = 1000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    # Compute the soft thresholding function
    return torch.sign(x) * torch.max(torch.abs(x) - shrinkage, torch.zeros(x.shape))

def ISTA(mu,shrinkage,K,y):
    # Identity matrix A and I
    A = torch.eye(y.shape[2])
    I = torch.eye(y.shape[2])
    # Initialize x
    x_out = torch.zeros(y.shape)
    for i in range(K):
        if i == 0:
            x_new = y
        else:
            x_new = x_out
        
        for j in range(x_new.shape[0]):
            x_out[j,:,:,:] = mu*torch.matmul(A,y[j,:,:,:]) + torch.matmul(I-mu*A*torch.transpose(A,0,1),x_new[j,:,:,:])
            x_new[j,:,:,:] = softthreshold(x_out[j,:,:,:],shrinkage)
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
MSE = torch.mean((x_clean_example - x_clean_pred) ** 2)
print('MSE: ', MSE)


# %% Entire dataset
# Run the algrithm for the entire dataset
x_clean_pred_train = torch.zeros(x_clean_train.shape)
x_clean_pred_test = torch.zeros(x_clean_test.shape)
for i in tqdm(range(x_clean_train.shape[0])):
    x_clean_pred_train[i] = ISTA(mu,shrinkage,K,x_noisy_train[i])
for i in tqdm(range(x_clean_test.shape[0])):
    x_clean_pred_test[i] = ISTA(mu,shrinkage,K,x_noisy_test[i])

# %% MSE
# Compute the MSE for the training and test set
mse_train = torch.mean((x_clean_pred_train - x_clean_train) ** 2)
mse_test = torch.mean((x_clean_pred_test - x_clean_test) ** 2)
print("MSE train: ", mse_train)
print("MSE test: ", mse_test)


# %% plot results
# Plot the noisy and the predicted clean image
# show the examples in a plot
plt.figure(figsize=(12,3))
plt.subplot(3,10,6)
plt.title("Noisy Examples",loc='right')
for i in range(10):
    plt.subplot(3,10,i+1)
    plt.imshow(x_noisy_train[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.subplot(3,10,16)
plt.title("Predicted Examples",loc='right')
for i in range(10):
    plt.subplot(3,10,i+11)
    plt.imshow(x_clean_pred_train[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.subplot(3,10,26)
plt.title("Clean Examples",loc='right')
for i in range(10):
    plt.subplot(3,10,i+21)
    plt.imshow(x_clean_train[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])    
plt.tight_layout()
plt.savefig("data_examples_results.png",dpi=300,bbox_inches='tight')
plt.show()


# %%
