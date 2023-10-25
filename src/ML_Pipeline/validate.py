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
from ML_Pipeline.network import UNetPP, VGGBlock
from ML_Pipeline.dataset import DataSet
from ML_Pipeline.train import train

# Define a validation function
def validate(deep_sup, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    # Switch the model to evaluation mode
    model.eval()

    # Determine the device to run the validation on (GPU if available, else CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Disable gradient computation during validation
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader)  # Create a progress bar
        for input, target, _ in val_loader:
            input = input.to(device)
            target = target.to(device)

            # Compute the output
            if deep_sup:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)

            # Update the average meters with loss and IoU values
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            # Create a dictionary of values to display in the progress bar
            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()  # Close the progress bar

    # Return a dictionary of average loss and IoU values
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)]
