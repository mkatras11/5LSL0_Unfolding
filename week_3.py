# %% 
# Importing the libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from Fast_MRI_dataloader import create_dataloaders
from tqdm import tqdm
from torch.fft import fft2, fftshift, ifft2, ifftshift

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Build the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32,32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 1, 3, padding=1)
        self.relu = nn.ReLU()
        self.batch_1 = nn.BatchNorm2d(16)
        self.batch_2 = nn.BatchNorm2d(32)
        self.batch_3 = nn.BatchNorm2d(32)
        self.batch_4 = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    # Forward propagation
    def forward(self, x):
        out = self.conv1(x)
        out = self.batch_1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batch_2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.batch_3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.batch_4(out)
        out = self.sigmoid(out)
        return out

# Calculate the validation loss
def validation_loss(model, test_loader, criterion, device):
    # Set the model to evaluation mode
    model.eval()
    # Initialize the validation loss
    validation_loss = 0
    # Initialize the number of validation steps
    validation_steps = 0
    # Loop over the test dataset
    for i, (pkspace, M, gt) in enumerate(tqdm(test_loader)):
        # Move to device 
        pkspace = pkspace.to(device)
        M = M.to(device)
        gt = gt.to(device)
        # Unsqueeze the kspace
        pkspace = pkspace.unsqueeze(1)
        # Unsqueeze the gt
        gt = gt.unsqueeze(1)
        # Get accelerated MRI image from partial kspace
        image = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(pkspace), norm="forward"))
        # Get the image from the model
        outputs = model(image)
        # Calculate the loss
        loss = criterion(outputs, gt)
        # Add the loss to the validation loss
        validation_loss += loss.item()
        # Increment the number of validation steps
        validation_steps += 1
    # Calculate the average validation loss
    loss = validation_loss / validation_steps
    # Return the validation loss
    return loss

# Train the model
def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs):
    # Set the model to training mode
    model.train()
    # Initialize the training loss and validation loss
    train_losses = []
    val_losses = []
    # Iterate over the epochs
    for epoch in range(epochs):
        # Initialize the training loss
        train_loss = 0
        # Initialize the validation loss
        val_loss = 0
        # Initialize the number of training steps
        train_steps = 0
        # Initialize the number of validation steps
        val_steps = 0
        # Loop over the training dataset
        for i, (pkspace, M, gt) in enumerate(tqdm(train_loader)):
            # Move to device 
            pkspace = pkspace.to(device)
            M = M.to(device)
            gt = gt.to(device)
            # Unsqueeze the kspace
            pkspace = pkspace.unsqueeze(1)
            # Unsqueeze the gt
            gt = gt.unsqueeze(1)
            # Get accelerated MRI image from partial kspace
            image = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(pkspace), norm="forward"))
            # Get the image from the model
            outputs = model(image)
            # Calculate the loss
            loss = criterion(outputs, gt)
            # Zero the gradients
            optimizer.zero_grad()
            # Backpropagate the loss
            loss.backward()
            # Update the weights
            optimizer.step()
            # Add the loss to the training loss
            train_loss += loss.item()
            # Increment the number of training steps
            train_steps += 1
        # Calculate the average training loss
        train_loss = train_loss / train_steps
        # Append the training loss to the training loss list
        train_losses.append(train_loss)
        # Calculate the validation loss
        val_loss = validation_loss(model, test_loader, criterion, device)
        # Append the validation loss to the validation loss list
        val_losses.append(val_loss)
        # Print the training and validation loss
        print("Epoch: {}/{} ".format(epoch+1, epochs),"Training Loss: {:.3f} ".format(train_loss), "Validation Loss: {:.3f}".format(val_loss))
    # Save the model
    torch.save(model.state_dict(), 'model.pt')
    # Return the training and validation loss lists
    return train_losses, val_losses

# Define plot loss for epochs
def plot_loss(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    # Plot the training loss
    plt.plot(epochs, train_losses, label="Training loss")
    # Plot the validation loss
    plt.plot(epochs, val_losses, label="Validation loss")
    # Set the x-axis label
    plt.xlabel("Epochs")
    # Set the y-axis label
    plt.ylabel("Loss")
    # Set the title
    plt.title("Training and validation loss")
    # Add the legend
    plt.legend()
    # Show the plot
    plt.show()


# Set the parameters of the model 
batch_size = 32
epochs = 10
learning_rate = 0.001

# Load the data
train_loader, test_loader = create_dataloaders('Fast_MRI_Knee', batch_size=batch_size)

# Initialize the model
model = CNN().to(device)

# Initialize the criterion
criterion = nn.MSELoss()

# Initialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_loss, val_loss = train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs)

# Plot the loss
plot_loss(train_loss, val_loss)


# %%
def plot_ex5c(test_acc_mri, test_x_out, test_gt, save_path):

    plt.figure(figsize = (10, 6))
    for i in range(5):
        plt.subplot(3,5,i+1)
        plt.imshow(test_acc_mri[i+1,0,:,:],cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 2:
            plt.title('Accelerated MRI')

        plt.subplot(3,5,i+6)
        plt.imshow(test_x_out[i+1,0,:,:],vmax=2,cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 2:
            plt.title('Reconstruction from CNN')

        plt.subplot(3,5,i+11)
        plt.imshow(test_gt[i+1,0,:,:],cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 2:
            plt.title('Ground truth')

    plt.savefig(f"{save_path}", dpi=300, bbox_inches='tight')
    plt.show()

# %% 
# Load the model saved model 
model = CNN().to(device)
model.load_state_dict(torch.load('model.pt'))
for i, (pkspace,M,gt) in enumerate(tqdm(test_loader)):
    if i == 1:
        break
image = torch.abs(ifft2(ifftshift(pkspace, dim=(1, 2)), dim=(1, 2)))
# Unsqueeze the image 
image = image.unsqueeze(1).to(device)
# Unsqueeze the gt
gt = gt.unsqueeze(1).to(device)
# Get the image from the model
outputs = model(image)
# Detach the outputs
outputs = outputs.detach().cpu().numpy()
image = image.detach().cpu().numpy()
gt = gt.detach().cpu().numpy()
# Print the output with the plot_ex5c function
plot_ex5c(image, outputs, gt, 'ex5c.png')
# %%
