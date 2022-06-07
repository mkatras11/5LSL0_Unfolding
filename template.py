# %% imports
# libraries
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
# local imports
import MNIST_dataloader

# %% set torches random seed
torch.random.manual_seed(0)

# %% preperations
# define parameters
data_loc = os.path.dirname(os.path.realpath(__file__))
batch_size = 64

mu = 1e-2
shrinkage = 1e-4
K = 100

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
    # Compute the ISTA with step size mu and shrinkage parameter shrinkage for K iterations
    x = torch.normal(0,1,size = y.shape)
    for k in range(K):
        x = softthreshold(x - y,shrinkage)
        x = x + (x - y) / (torch.norm(x - y) ** 2 + mu)
    return x


x_clean_pred = ISTA(mu,shrinkage,K,x_noisy_example)
print(x_clean_pred.shape)

# Run the algrithm for the entire dataset
x_clean_pred_train = torch.zeros(x_clean_train.shape)
x_clean_pred_test = torch.zeros(x_clean_test.shape)
for i in tqdm(range(x_clean_train.shape[0])):
    x_clean_pred_train[i] = ISTA(mu,shrinkage,K,x_noisy_train[i])
for i in tqdm(range(x_clean_test.shape[0])):
    x_clean_pred_test[i] = ISTA(mu,shrinkage,K,x_noisy_test[i])


# %% plot results
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
plt.savefig("data_examples_results.png",dpi=300,bbox_inches='tight')
plt.show()


    



   
