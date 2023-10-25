import os
import cv2
import numpy as np
import torch.utils.data

class DataSet(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, transform=None):
        """
        Custom dataset for image segmentation.

        Args:
            img_ids (list): List of image file names (without extensions).
            img_dir (str): Directory containing the image files.
            mask_dir (str): Directory containing the mask files.
            img_ext (str): Image file extension (e.g., '.jpg').
            mask_ext (str): Mask file extension (e.g., '.png').
            transform (callable, optional): Optional data augmentation and preprocessing transformations.
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.img_ids)

    def __getitem__(self, idx):
        """
        Returns a sample (image and mask) and associated metadata.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image, mask, and metadata dictionary.
        """
        img_id = self.img_ids[idx]

        # Load the image
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        # Load and stack the mask (in grayscale) into a single channel
        mask = []
        mask.append(cv2.imread(os.path.join(self.mask_dir, img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)

        # Apply data augmentation and preprocessing transformations
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # Normalize and transpose the image and mask tensors
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)  # Channels-first format
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)  # Channels-first format

        # Return the image, mask, and associated metadata
        return img, mask, {'img_id': img_id}
