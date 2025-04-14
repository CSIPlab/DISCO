import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DIV2KDataset(Dataset):
    def __init__(self, img_dir, train=True, num_imgs = -1):
        """
        Args:
            img_dir (str): Path to the image directory.
            train (bool): Whether the dataset is for training or testing.
        """
        self.img_dir = img_dir
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.train = train
        self.num_imgs = num_imgs
        
        # Define transformations
        self.train_transforms = transforms.Compose([
            # transforms.RandomResizedCrop(512),
            transforms.Resize(512),         # 
            transforms.RandomCrop(256),     # Crop the center 512x512
            # transforms.CenterCrop(256),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        
        self.test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256), 
            transforms.ToTensor()
        ])

    def __len__(self):
        return self.num_imgs if self.num_imgs > -1 else len(self.img_files)

    def __getitem__(self, idx):
        # idx = 3
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert("RGB")
        
        if self.train:
            image = self.train_transforms(image)
        else:
            image = self.test_transforms(image)

        return image, self.img_files[idx]

# Set up the data loader
def get_dataloader(img_dir, batch_size=32, train=True, shuffle=True, num_imgs = -1, **kwargs):
    dataset = DIV2KDataset(img_dir, train=train, num_imgs = num_imgs, )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle if train else False)
