import os
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio  # Assuming MAT files are loaded using scipy.io
import numpy as np
from torchvision import transforms
import random
import torchvision.transforms.functional as TF

from .sidd_medium_loader import SIDD_Dataset
class ImageDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (str): Path to the root directory containing data folders.
        """
        self.root_dir = root_dir
        self.data_paths = self._get_data_paths()

    def _get_data_paths(self):
        """Fetch all data folder paths in the root directory."""
        folders = [os.path.join(self.root_dir, folder) for folder in os.listdir(self.root_dir) 
                   if os.path.isdir(os.path.join(self.root_dir, folder))]
        data_paths = []
        for folder in folders:
            gt_path = os.path.join(folder, next(f for f in os.listdir(folder) if f.startswith("GT_RAW_")))
            noisy_path = os.path.join(folder, next(f for f in os.listdir(folder) if f.startswith("NOISY_RAW_")))
            metadata_path = os.path.join(folder, next(f for f in os.listdir(folder) if f.startswith("METADATA_RAW_")))
            image_id = os.path.basename(folder)
            data_paths.append((image_id, gt_path, noisy_path, metadata_path))
        return data_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        image_id, gt_path, noisy_path, metadata_path = self.data_paths[idx]

        # Load .MAT files
        gt_image = self._load_mat(gt_path)
        noisy_image = self._load_mat(noisy_path)
        metadata = self._load_mat(metadata_path)

        return image_id, gt_image, noisy_image, metadata

    def _load_mat(self, file_path):
        """Load and return data from a .MAT file."""
        mat_data = sio.loadmat(file_path)
        # Assuming the data key is the first variable in the .MAT file
        # Adjust as necessary for your .MAT file structure
        return next(iter(mat_data.values()))

class ToTensorTransform:
    def __call__(self, sample):
        noisy_image, gt_image = sample
        return TF.to_tensor(noisy_image), TF.to_tensor(gt_image)
    
class PairedTransform:
    def __init__(self, rotation=10):
        self.rotation = rotation

    def __call__(self, sample):
        noisy_image, gt_image = sample

        # Random horizontal flip
        if random.random() > 0.5:
            noisy_image = TF.hflip(noisy_image)
            gt_image = TF.hflip(gt_image)

        # Random vertical flip
        if random.random() > 0.5:
            noisy_image = TF.vflip(noisy_image)
            gt_image = TF.vflip(gt_image)

        # Random rotation
        angle = random.uniform(-self.rotation, self.rotation)
        noisy_image = TF.rotate(noisy_image, angle)
        gt_image = TF.rotate(gt_image, angle)

        return noisy_image, gt_image

def get_dataloader(dataset_root, mode = 'train', crop_size = 256, batch_size = 4, shuffle = True, num_imgs = -1, **kwargs):
    
    # # Directory where the dataset resides
    # dataset_root = "/data/SalmanAsif/nyism/SSID/medium/SIDD_Medium_Srgb/Data/"
    # mode = 'train'

    # dataset_root = "/data/SalmanAsif/nyism/SSID/validation/"
    # mode = 'val'

    # Define transformations (e.g., resizing, normalization, etc.)
    if mode == 'train':
        data_transforms = transforms.Compose([
            # transforms.RandomHorizontalFlip(), 
            # transforms.RandomVerticalFlip(), 
            # transforms.RandomRotation(15),  
            # PairedTransform(),
            ToTensorTransform(), # Convert PIL image to tensor
        ])
    else:
       data_transforms = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL image to tensor
        ])
    # Create dataset and data loader
    sidd_dataset = SIDD_Dataset(root_dir=dataset_root, transform=data_transforms, crop_size = crop_size, mode = mode, num_imgs = num_imgs)
    
    return DataLoader(sidd_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)

if __name__ == "__main__":
    # # Define the dataset and DataLoader
    root_dir = "/home/yismaw/Data/"
    dataset = ImageDataset(root_dir)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Example usage
    for image_id, gt_image, noisy_image, metadata in data_loader:
        print("Image ID:", image_id)
        print("Ground Truth Image Shape:", gt_image.shape)
        print("Noisy Image Shape:", noisy_image.shape)
        print("Metadata:", metadata)
        break
