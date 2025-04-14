import random 

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import crop, resize


import scipy.io as sio

class SIDD_Dataset(Dataset):
    def __init__(self, root_dir, transform=None, mode="train", crop_size=512, num_imgs = -1):
        """
        Args:
            root_dir (string): Directory with all the data.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (string): Either 'train' (default) or 'val'.
            crop_size (int): Size of the square crop for training images. Default is 512.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.val_transform = transforms.Compose([transforms.ToTensor()])
        self.mode = mode
        self.crop_size = crop_size
        self.num_imgs = num_imgs
        if mode == "train":
            self.data_pairs = self._prepare_data_pairs()
        elif mode == "val":
            self.val_noisy, self.val_gt = self._load_val_data()
            
        print("SIDD Medium Dataset: ", self.num_imgs, self.crop_size)    

    def _prepare_data_pairs(self):
        data_pairs = []
        for scene_dir in os.listdir(self.root_dir):
            scene_path = os.path.join(self.root_dir, scene_dir)
            if os.path.isdir(scene_path):
                noisy_images = sorted(
                    [os.path.join(scene_path, f) for f in os.listdir(scene_path) if "NOISY_SRGB" in f]
                )
                gt_images = sorted(
                    [os.path.join(scene_path, f) for f in os.listdir(scene_path) if "GT_SRGB" in f]
                )
                for noisy, gt in zip(noisy_images, gt_images):
                    data_pairs.append((noisy, gt))
        return data_pairs

    def _load_val_data(self):
        noisy_mat_path = os.path.join(self.root_dir, "ValidationNoisyBlocksSrgb.mat")
        gt_mat_path = os.path.join(self.root_dir, "ValidationGtBlocksSrgb.mat")

        noisy_data = sio.loadmat(noisy_mat_path)["ValidationNoisyBlocksSrgb"]
        gt_data = sio.loadmat(gt_mat_path)["ValidationGtBlocksSrgb"]

        # Flatten the dimensions: (imgs, blocks, H, W, C) -> (imgs*blocks, H, W, C)
        imgs, blocks, H, W, C = noisy_data.shape
        
        return noisy_data.reshape(-1, H, W, C), gt_data.reshape(-1, H, W, C)

    def __len__(self):
        if self.num_imgs > -1:
            return self.num_imgs

        if self.mode == "train":
            return len(self.data_pairs)
        elif self.mode == "val":
            # Length of flattened validation dataset
            return self.val_noisy.shape[0]

    def __getitem__(self, idx):
        if self.mode == "train":
            noisy_path, gt_path = self.data_pairs[idx]
            noisy_image = Image.open(noisy_path).convert("RGB")
            gt_image = Image.open(gt_path).convert("RGB")

            width, height = noisy_image.size
            top = torch.randint(self.crop_size, height - self.crop_size + 1, (1,)).item()
            left = torch.randint(self.crop_size, width - self.crop_size + 1, (1,)).item()
            
            noisy_image = crop(noisy_image, top, left, self.crop_size, self.crop_size)
            gt_image = crop(gt_image, top, left, self.crop_size, self.crop_size)
            
            if self.transform:
                noisy_image, gt_image = self.transform((noisy_image, gt_image))
                
        elif self.mode == "val":
            noisy_image = self.val_noisy[idx]
            gt_image = self.val_gt[idx]
            
            if self.val_transform: 
                noisy_image = self.val_transform(noisy_image)
                gt_image = self.val_transform(gt_image)

        return {"noisy": noisy_image, "gt": gt_image}


if __name__ == "__main__":    
    # Directory where the dataset resides
    dataset_root = "/data/SalmanAsif/nyism/SSID/medium/SIDD_Medium_Srgb/Data/"
    mode = 'train'

    dataset_root = "/data/SalmanAsif/nyism/SSID/validation/"
    mode = 'val'

    # Define transformations (e.g., resizing, normalization, etc.)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to tensor
    ])

    # Create dataset and data loader
    sidd_dataset = SIDD_Dataset(root_dir=dataset_root, transform=transform, crop_size = 256, mode = mode)
    data_loader = DataLoader(sidd_dataset, batch_size=4, shuffle=True, num_workers=0)

    # Example usage
    for batch in data_loader:
        noisy_batch = batch["noisy"]
        gt_batch = batch["gt"]
        
        print(f"noisy_batch : {noisy_batch.mean(), noisy_batch.max(), noisy_batch.min()}")
        print(f"gt_batch : {gt_batch.mean()}")
        break
