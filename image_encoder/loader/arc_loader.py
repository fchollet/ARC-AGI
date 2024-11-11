import os
import json
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader


# ========================================================================
# Easy calls for external use
# ========================================================================
def _get_data_fp(filename):
    """
        Function to return the filepath to our training data.
        Pass in the filename of the dataset you want to use - e.g., 'ibot_traindata_aggregate.parquet'
    """
    try:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_file_dir = os.getcwd()
    
    root = os.path.abspath(os.path.join(current_file_dir, '..'))
    file = os.path.join(root, 'train_data', filename)
    return file


def get_dataloader(filename, loader_params):
    """
        Function for easy external calling that returns our dataloader given loader params and a filename.

        Inputs:
            filename (str): Name of file we'll load our data from. Input to 'get_data_fp'
            loader_params (dict): Dictionary of our loader parameters
    """
    fp = _get_data_fp(filename)
    dataset = ARCDatasetWrapper(TxtDictDataset(fp), pad_images=loader_params['pad_images'], percent_mask=loader_params['percent_mask'])
    dataloader = ARCDataLoader(dataset, batch_size=loader_params['batch_size'], shuffle=loader_params['shuffle'])
    return dataloader


# ========================================================================
# Custom Dataset, Dataset Wrapper, and DataLoader
# ========================================================================
class TxtDictDataset(Dataset):
    def __init__(self, fp):
        """
        Custom dataset to load in a .txt file of dict objects.
        Returns a tuple of (key, tensor).
        """
        self.fp = fp
        self.samples = self.load_samples(fp)

    def load_samples(self, fp):
        samples = []
        with open(fp, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                for key, value in entry.items():
                    tensor = torch.tensor(value, dtype=torch.int32)
                    samples.append((key, tensor))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



class ARCDatasetWrapper:
    def __init__(self, dataset, pad_images=False, percent_mask=0.1, pad_to_32x32=False):
        """
        Wrapper for TxtDictDataset to apply additional processing.
        
        Parameters:
        - dataset (Dataset): The base dataset to wrap, which should yield tuples of (key, image).
            * key (str): Identifier for the data sample.
            * image (torch.Tensor): 2D tensor representing the image, expected to be in shape (H, W).
        - pad_images (bool): If True, pads images to 32x32 using pad_to_32x32 function.
        - percent_mask (float): Probability of each pixel in the mask being set to 1.
        """
        self.dataset = dataset
        self.pad_images = pad_images
        self.percent_mask = percent_mask
        self.crop_args = {
            'min_crop_shape': 3,
            'max_crop': 5
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        key, images = self.dataset[idx]        
        image_masks = self.get_image_mask(images)
        
        transformed_images = images.clone()
        transformed_images = self.shuffle_colors(transformed_images)
        transformed_image_masks = self.get_image_mask(transformed_images)
        # transformed_image = self.global_crop(transformed_image)
        if self.pad_images:
            images = self.pad_to_32x32(images)
            image_masks = self.pad_to_32x32(image_masks)
            transformed_images = self.pad_to_32x32(transformed_images)
            transformed_image_masks = self.pad_to_32x32(transformed_image_masks)
        return key, images, image_masks, transformed_images, transformed_image_masks

    def pad_to_32x32(self, image):
        """Pads the input image tensor to 32x32 with custom padding rules."""        
        height, width = image.shape
        if height == 32 and width == 32:
            return image  # No padding needed if already 32x32
        
        # Add '11' padding on the right and bottom edges
        padded_image = torch.nn.functional.pad(image, (0, 1, 0, 1), value=11)
        
        # Fill remaining space to reach 32x32 with '12'
        pad_bottom = max(0, 32 - padded_image.shape[0])
        pad_right = max(0, 32 - padded_image.shape[1])
        final_padded_image = torch.nn.functional.pad(padded_image, (0, pad_right, 0, pad_bottom), value=12)
        
        return final_padded_image

    def shuffle_colors(self, image):
        """Apply a random color mapping to shuffle integers from 0 to 9 in place, avoiding double-mapping issues."""
        original_values = list(range(10))
        shuffled_values = torch.randperm(10).tolist()
        mapping = {original: shuffled for original, shuffled in zip(original_values, shuffled_values)}
        mapped_image = image.clone()    # Create clone to prevent double mapping
        for original_value, new_value in mapping.items():
            mapped_image[image == original_value] = new_value
        return mapped_image

    def global_crop(self, image):
        """Apply a global cropping transformation based on specified conditions."""
        height, width = image.shape
        if width > self.crop_args['min_crop_shape']:
            max_width_crop = min(self.crop_args['max_crop'], int((width * 1.5) / self.crop_args['max_crop']))
            crop_width = random.randint(0, max_width_crop)
            i = random.randint(0, crop_width)
            j = width - crop_width
            image = image[:, i:j]
    
        if height > self.crop_args['min_crop_shape']:
            max_height_crop = min(self.crop_args['max_crop'], int((height * 1.5) / self.crop_args['max_crop']))
            crop_height = random.randint(0, max_height_crop)
            i = random.randint(0, crop_height)
            j = height - crop_height
            image = image[i:j, :]
    
        return image

    def get_image_mask(self, image):
        """Generate an image mask with values 0 and 1 based on the percent_mask probability."""
        height, width = image.shape
        mask = torch.bernoulli(torch.full((height, width), self.percent_mask)).int()
        if self.pad_images:
            pad_bottom = max(0, 32 - height)
            pad_right = max(0, 32 - width)
            mask = torch.nn.functional.pad(mask, (0, pad_right, 0, pad_bottom), value=0)
        return mask



class ARCDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, collate_fn=None):
        if collate_fn is None:
            collate_fn = self.custom_collate_fn
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    @staticmethod
    def custom_collate_fn(batch):
        keys = [item[0] for item in batch]
        images = [item[1] for item in batch]
        image_masks = [item[2] for item in batch]
        transformed_images = [item[3] for item in batch]
        transformed_image_masks = [item[4] for item in batch]
        return keys, torch.stack(images), torch.stack(image_masks), torch.stack(transformed_images), torch.stack(transformed_image_masks)