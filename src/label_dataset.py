import os
import glob
import torch
from torch.utils.data import Dataset
from .utils import load_image

class LabeledDataset(Dataset):
    """
    Dataset for paired synthetic data and labels.
    This will be used to train a generator that can produce realistic-looking
    synthetic data while preserving the label information.
    """
    def __init__(self, synthetic_dir, labels_dir, transform=None):
        self.synthetic_paths = sorted(glob.glob(os.path.join(synthetic_dir, '*')))
        self.labels_paths = sorted(glob.glob(os.path.join(labels_dir, '*')))
        self.transform = transform
        
        # Ensure we have matching pairs
        assert len(self.synthetic_paths) == len(self.labels_paths), \
            f"Mismatch between synthetic data ({len(self.synthetic_paths)}) and labels ({len(self.labels_paths)})"

    def __len__(self):
        return len(self.synthetic_paths)

    def __getitem__(self, idx):
        synthetic_path = self.synthetic_paths[idx]
        label_path = self.labels_paths[idx]
        
        # Extract base names to verify matching pairs
        synthetic_name = os.path.basename(synthetic_path)
        label_name = os.path.basename(label_path)
        
        # Verify that the files are a matching pair (by index)
        synthetic_idx = int(synthetic_name.split('_')[-1].split('.')[0])
        label_idx = int(label_name.split('_')[-1].split('.')[0])
        assert synthetic_idx == label_idx, f"Mismatched indices: {synthetic_idx} vs {label_idx}"
        
        synthetic = load_image(synthetic_path)
        label = load_image(label_path)
        
        if self.transform:
            synthetic = self.transform(synthetic)
            # For labels, we might want a different transform that preserves the label values
            label = self.transform(label)
            
        return {'synthetic': synthetic, 'label': label}
