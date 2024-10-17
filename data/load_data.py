import os
import json
import torch
from torch.utils.data import Dataset

TRAIN_DIR = 'training'
EVAL_DIR = 'evaluation'


def load_dataset(mode='evaluation'):
    """ mode: 'training' or 'evaluation'. Determines which directory to load from. """
    return JSONDataset(mode)


class JSONDataset(Dataset):
    """ 
        Loads in data and returns it as nested dictionary of following format: 
            {
                'train': [{'input': torch.tensor, 'output': torch.tensor}, ...],
                'test': [{'input': torch.tensor, 'output': torch.tensor}, ...]
            }
        Where each tensor is a 3D tensor representing a [1 x h x w] grid. Reason for 3D (vs. 2D) is a PyTorch quirk when doing nested data like this.
    """
    
    def __init__(self, mode='evaluation'):
        """ mode: 'training' or 'evaluation'. Determines which directory to load from. """
        if mode not in ['training', 'evaluation']:
            raise ValueError("mode must be 'training' or 'evaluation'")
        
        # Get the current directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set the target directory based on the mode
        target_dir = os.path.join(script_dir, TRAIN_DIR if mode == 'training' else EVAL_DIR)
        
        # Ensure the directory exists
        if not os.path.exists(target_dir):
            raise FileNotFoundError(f"Directory {target_dir} not found")
        
        # Load JSON files into the dataset
        self.data = []
        for file_name in os.listdir(target_dir):
            if file_name.endswith('.json'):
                file_path = os.path.join(target_dir, file_name)
                with open(file_path, 'r') as f:
                    json_data = json.load(f)
                    json_data['train'] = [{'input': torch.tensor(sample['input']), 'output': torch.tensor(sample['output'])} for sample in json_data['train']]
                    json_data['test'] = [{'input': torch.tensor(sample['input']), 'output': torch.tensor(sample['output'])} for sample in json_data['test']]
                    self.data.append(json_data)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        return self.data[idx]