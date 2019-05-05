# import the usual resources
import matplotlib.pyplot as plt
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## TODO: Once you've define the network, you can instantiate it
# one example conv layer has been provided for you
from models import NaimishNet

net = NaimishNet().to(device)
print(net)

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
from data_load import FacialKeypointsDataset
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from data_load import Rescale, RandomCrop, Normalize, ToTensor


## TODO: define the data_transform using transforms.Compose([all tx's, . , .])
# order matters! i.e. rescaling should come before a smaller crop
data_transform = transforms.Compose([Rescale((250, 250)),
                                     RandomCrop((224, 224)),
                                     Normalize(),
                                     ToTensor()])

test_data_transform = transforms.Compose([Rescale((224, 224)),
                                          Normalize(),
                                          ToTensor()])

# testing that you've defined a transform
assert(data_transform is not None), 'Define a data_transform'

# create the transformed dataset
transformed_dataset = FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv',
                                             root_dir='data/training/',
                                             transform=data_transform)

test_dataset = FacialKeypointsDataset(csv_file='data/test_frames_keypoints.csv',
                                      root_dir='data/test/',
                                      transform=test_data_transform)


print('Number of images: ', len(transformed_dataset))

# iterate through the transformed dataset and print some stats about the first few samples
for i in range(4):
    sample = transformed_dataset[i]
    print(i, sample['image'].size(), sample['keypoints'].size())


# load training data in batches
batch_size = 8

train_loader = DataLoader(transformed_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=4)


## TODO: Define the loss and optimization
import torch.optim as optim

criterion = nn.SmoothL1Loss() #nn.MSELoss()

optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)


import sys
import math


def process_data(data_loader, train=False):
    if train:
        net.train()
    else:
        net.eval()

    losses = []
    average_loss = 0.0
    #running_loss = 0.0

    batch_count = len(data_loader)

    for batch_i, data in enumerate(data_loader):
        # get the input images and their corresponding labels
        images = data['image']
        key_pts = data['keypoints']

        # flatten pts
        key_pts = key_pts.view(key_pts.size(0), -1)

        # convert variables to floats for regression loss
        key_pts = key_pts.type(torch.cuda.FloatTensor)
        images = images.type(torch.cuda.FloatTensor)

        if train:
            # forward pass to get outputs
            output_pts = net(images)
        else:
            with torch.no_grad():
                output_pts = net(images)

        # calculate the loss between predicted and target keypoints
        loss = criterion(output_pts, key_pts)

        if train:
            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            # backward pass to calculate the weight gradients
            loss.backward()
            # update the weights
            optimizer.step()

        # print loss statistics
        # to convert loss into a scalar and add it to the running_loss, use .item()
        running_loss = loss.item()
        losses.append(running_loss)

        # average_training_loss = total_training_loss / (batch_i + 1)
        #if batch_i % 10 == 9:  # print every 10 batches
        action = "training" if train else "validating"
        print('\r{} batch: {}/{}, current loss: {:.3f}'.format(action, batch_i + 1, batch_count, running_loss), end="")
        sys.stdout.flush()
            #running_loss = 0.0

    average_loss = np.mean(losses)

    return average_loss


def train_net(n_epochs, patience):
    last_train_loss = math.inf
    last_valid_loss = math.inf
    best_valid_loss = last_valid_loss
    epochs_without_improvement = 0

    for epoch in range(n_epochs):
        print("--------------------------------------------------------------------------")
        print("Epoch: {}, last train loss: {:.3f}, last valid loss: {:.3f}, best valid loss: {:.3f}".format(
            epoch + 1, last_train_loss, last_valid_loss, best_valid_loss
        ))
        last_train_loss = process_data(train_loader, train=True)
        test_loss = process_data(test_loader, train=False)

        last_valid_loss = test_loss
        if last_valid_loss < best_valid_loss:
            best_valid_loss = last_valid_loss
            torch.save(net.state_dict(), "saved_models/keypoints_model.pth")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("\nstopping early")
            break


# train your network
n_epochs = 50
patience = 3

train_net(n_epochs, patience)
