import os
import glob
from torch.utils.data import Dataset
from .utils import load_image

class UnpairedDataset(Dataset):
    """
    Loads unpaired real and synthetic images for CycleGAN training.
    Works with grayscale images by default.
    """
    def __init__(self, real_dir, synthetic_dir, transform=None, grayscale=True):
        self.real_paths = sorted(glob.glob(os.path.join(real_dir, '*')))
        self.synthetic_paths = sorted(glob.glob(os.path.join(synthetic_dir, '*')))
        self.transform = transform
        self.grayscale = grayscale

    def __len__(self):
        return max(len(self.real_paths), len(self.synthetic_paths))

    def __getitem__(self, idx):
        real_path = self.real_paths[idx % len(self.real_paths)]
        syn_path = self.synthetic_paths[idx % len(self.synthetic_paths)]
        real = load_image(real_path, grayscale=self.grayscale)
        synthetic = load_image(syn_path, grayscale=self.grayscale)
        if self.transform:
            real = self.transform(real)
            synthetic = self.transform(synthetic)
        return {'real': real, 'synthetic': synthetic}
