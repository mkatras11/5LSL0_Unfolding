# %% imports
# libraries
from re import X
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# local imports
import MNIST_dataloader

class Lista (nn.Module):
    def __init__(self):
        super(Lista, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 3, padding= 1)
        self.unfolds = 3
        self.shrikage = nn.Parameter(0.01 * torch.ones(1,self.unfolds))
    
    def smooth_softthreshold(self,x,shrinkage):
        return x + 0.5 * ((torch.sqrt(((x - shrinkage)**2) + 1)) - (torch.sqrt(((x + shrinkage)**2) + 1)))

    def forward(self,x):
       for i in range(self.unfolds):
            if i == 0:
                input = 0
            else:
                input = x_conv2

            x_conv1 = self.conv1(x) + input      
            x_out = self.smooth_softthreshold(x_conv1,self.shrikage[:,i])
            x_conv2 = self.conv1(x_out)
      
            return x_out

def Trainer(model, train_loader, optimizer, criterion, epochs):
    model.train()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loss = 0.0
    loss_list = []

    for epoch in range(epochs):
        # Create a progress bar using TQDM
        sys.stdout.flush()
        with tqdm(total=len(train_loader), desc=f'Training') as pbar:
            for batch_idx, (data_clean, data_noisy, labels) in enumerate(train_loader):
                data_clean, data_noisy, labels = data_clean.to(DEVICE), data_noisy.to(DEVICE), labels.to(DEVICE)
                model.to(DEVICE)
                x_out = model(data_noisy)
                loss = criterion(x_out, data_clean)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update the progress bar
                pbar.update(list(data_noisy.shape)[0])

                train_loss += loss.item()
        sys.stdout.flush()

        train_loss /= len(train_loader.dataset)
        loss_list.append(train_loss)
        print(f'Epoch {epoch+1} Loss: {train_loss}')
        train_loss = 0.0

    return loss_list
    
def Test(model, test_loader, criterion):
    model.eval()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loss = 0.0
    loss_list = []
    # Create a progress bar using TQDM
    sys.stdout.flush()
    with torch.no_grad(), tqdm(total=len(test_loader), desc=f'Test') as pbar:
        for batch_idx, (data_clean, data_noisy, labels) in enumerate(test_loader):
            data_clean, data_noisy, labels = data_clean.to(DEVICE), data_noisy.to(DEVICE), labels.to(DEVICE)
            x_out = model(data_noisy)
            loss = criterion(x_out, data_clean)

            # Update the progress bar
            pbar.update(list(data_noisy.shape)[0])

            test_loss += loss.item()
    sys.stdout.flush()

    test_loss /= len(test_loader.dataset)
    loss_list.append(test_loss)
    print(f'Test Loss: {test_loss}')

    return loss_list

if __name__ == '__main__':
    # set torches random seed
    torch.random.manual_seed(0)

    model = Lista()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-5)
    data_loc = os.path.dirname(os.path.realpath(__file__))
    batch_size = 64
    train_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

    # Train the network
    print("Starting training process...")
    train_loss = Trainer(model, train_loader, optimizer, criterion, epochs=10)   
    print("Finished training.")
    save_path = 'model_lista.pth'
    torch.save(model.state_dict(), save_path)
    print("Saved trained model as {}.".format(save_path))

    # Test the network
    print("Starting testing process...")
    save_path = 'model_lista.pth'
    model.load_state_dict(torch.load(save_path))
    test_loss = Test(model, test_loader, criterion)
    print("Finished testing.")
    
    epochs = np.arange(1, len(train_loss) + 1)
    # Accuracy and loss plots
    plt.figure(figsize=(15, 10))
    plt.plot(epochs, train_loss)
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train loss'], loc='upper right')
    plt.show()
    

        