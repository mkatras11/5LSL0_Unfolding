# %% 
# Importing the libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from Fast_MRI_dataloader import create_dataloaders
from tqdm import tqdm
#from week_2 import pkspace_to_image , image_to_kspace, image_to_partialkspace

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Build the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64,32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 1, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.sigmoid = nn.Sigmoid()

    # Forward propagation
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.sigmoid(self.conv4(x))
        return x

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
        loss = train_loss / train_steps
        # Append the training loss to the training loss list
        train_losses.append(loss)
        # Calculate the validation loss
        loss = validation_loss(model, test_loader, criterion, device)
        # Append the validation loss to the validation loss list
        val_losses.append(loss)
        # Print the training and validation loss
        print("Epoch: {}/{}.. ".format(epoch+1, epochs),"Training Loss: {:.3f}.. ".format(loss), "Validation Loss: {:.3f}".format(loss))
        # Save the model
        torch.save(model.state_dict(), 'model_epoch_' + str(epoch+1) + '.pt')
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
