import os
import yaml
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from glob import glob
from tqdm import tqdm
from collections import OrderedDict
from ML_Pipeline.utils import AverageMeter, iou_score
from albumentations import Resize
from albumentations.augmentations import transforms
from sklearn.model_selection import train_test_split
from albumentations.core.composition import Compose, OneOf
from albumentations.augmentations.transforms import RandomRotate90
from ML_Pipeline.network import UNetPP
from ML_Pipeline.dataset import DataSet

# Define a training function that takes input arguments
def train(deep_sup, train_loader, model, criterion, optimizer):
    # Initialize average meters to keep track of loss and IoU
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    # Set the model to training mode
    model.train()

    # Initialize a progress bar to track training progress
    pbar = tqdm(total=len(train_loader))

    # Determine the device to run the training on (GPU if available, else CPU)
    device = torch.device("cuda:0" if torch.cuda.is available() else "cpu")

    # Iterate over the training data
    for input, target, _ in train_loader:
        # Move input and target data to the specified device (GPU or CPU)
        input = input.to(device)
        target = target.to(device)

        # Compute the output
        if deep_sup:
            # If using deep supervision, compute outputs at multiple scales
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            # If not using deep supervision, compute a single output
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)

        # Compute gradients and update model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update average meters with loss and IoU values
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        # Create a dictionary of values to display in the progress bar
        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    
    # Close the progress bar
    pbar.close()

    # Return a dictionary of average loss and IoU values
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)]
