# %% 
# Importing the libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from Fast_MRI_dataloader import create_dataloaders
from tqdm import tqdm
from week_2 import pkspace_to_image , image_to_kspace, image_to_partialkspace

# Load the data
train_loader, test_loader = create_dataloaders('Fast_MRI_Knee', batch_size=2)

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
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
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
        kspace = kspace.to(device)
        M = M.to(device)
        gt = gt.to(device)
        # Unsqueeze the kspace
        kspace = kspace.unsqueeze(1)
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
    validation_loss = validation_loss / validation_steps
    # Return the validation loss
    return validation_loss

# Train the model
def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs):
    # Set the model to training mode
    model.train()
    # Initialize the training loss
    training_loss = 0
    # Initialize the number of training steps
    training_steps = 0
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
        # Add the loss to the training loss
        training_loss += loss.item()
        # Zero the gradients
        optimizer.zero_grad()
        # Backpropagate the loss
        loss.backward()
        # Update the weights
        optimizer.step()
        # Increment the number of training steps
        training_steps += 1
    # Calculate the average training loss
    training_loss = training_loss / training_steps
    # Calculate the validation loss
    validation_loss = validation_loss(model, test_loader, criterion, device)
    # Print the training loss and validation loss
    print("Training loss: {}, Validation loss: {}".format(training_loss, validation_loss))
    # Return the training loss and validation loss
    return training_loss, validation_loss

# Initialize the model
model = CNN().to(device)

# Set the parameters of the model 
batch_size = 32
epochs = 10
learning_rate = 0.001

# Initialize the criterion
criterion = nn.MSELoss()

# Initialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
training_loss, validation_loss = train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs)




