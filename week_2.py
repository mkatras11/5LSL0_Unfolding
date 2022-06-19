import torch
from Fast_MRI_dataloader import create_dataloaders
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import tqdm

# define parameters
data_loc = os.path.dirname(os.path.realpath(__file__)) #change the datalocation to something that works for you
data_loc = os.path.join(data_loc,'Fast_MRI_Knee')
batch_size = 2

# create dataloaders
train_loader, test_loader = create_dataloaders(data_loc, batch_size)

# go over the dataset
for i,(kspace, M, gt) in enumerate(tqdm(test_loader)):
    continue

