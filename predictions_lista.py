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
from model_lista import Lista



# %% preperations
# define parameters
data_loc = os.path.dirname(os.path.realpath(__file__))
batch_size = 64

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

save_path = 'model_lista.pth'
model = Lista()
model = model.cpu()
model.load_state_dict(torch.load(save_path))
model.eval()
model_clean_pred = model(x_noisy_example)

# Transform torch to numpy
if torch.is_tensor(model_clean_pred):
  model_clean_pred = model_clean_pred.detach()
  model_clean_pred = model_clean_pred.cpu().numpy()

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
    plt.imshow(model_clean_pred[i,0,:,:],cmap='gray')
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