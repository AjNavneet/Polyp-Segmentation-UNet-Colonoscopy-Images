import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        # Initialize the AverageMeter object
        self.reset()

    def reset(self):
        # Reset the values to their initial state
        self.val = 0   # Current value
        self.avg = 0   # Running average
        self.sum = 0   # Sum of values
        self.count = 0  # Count of data points

    def update(self, val, n=1):
        # Update the values with a new data point
        self.val = val  # Set the current value to the provided value
        self.sum += val * n  # Add the value times 'n' to the sum
        self.count += n  # Increment the count by 'n'
        self.avg = self.sum / self.count  # Recalculate the running average

def iou_score(output, target):
    smooth = 1e-5  # A small constant to avoid division by zero

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()  # Apply sigmoid to the output and convert to a NumPy array
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()  # Convert the target tensor to a NumPy array

    output_ = output > 0.5  # Threshold the output to create a binary mask
    target_ = target > 0.5  # Threshold the target to create a binary mask

    # Calculate the intersection and union of the binary masks
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    # Compute the Intersection over Union (IoU) score with smoothing
    return (intersection + smooth) / (union + smooth)
