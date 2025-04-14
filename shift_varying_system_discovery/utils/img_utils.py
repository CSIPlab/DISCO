from PIL import Image
import numpy as np
import torch 
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

# def extract_patches(image, f, stride = 3):
#     # Padding the image to handle borders (apply padding for each channel)
#     padded_image = F.pad(image, (f, f, f, f), mode='reflect')
    
#     patch_size = 2 * f + 1
#     # Extract patches using unfold for each channel separately
#     patches = padded_image.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
#     # Reshape patches to be easier to work with
#     patches = patches.contiguous().view(padded_image.size(0), -1, patch_size, patch_size)
#     return patches

def patchify_img(img, patch_size):
    assert len(img.shape) == 4  # Ensure img has 4 dimensions [B, C, H, W]
    
    # Calculate padding for height and width to make dimensions divisible by patch_size
    pad_h = (patch_size - img.shape[2] % patch_size) % patch_size
    pad_w = (patch_size - img.shape[3] % patch_size) % patch_size
    
    # Apply padding symmetrically around height and width
    padded_img = F.pad(img, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), mode='reflect')
    
    return rearrange(padded_img, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size)

def patch_to_img(patches, patch_size, img_size):
    original_size = (img_size, img_size)
    h = ((patch_size - img_size % patch_size) % patch_size + img_size) // patch_size
    
    # Rearrange patches back to the padded image shape
    padded_img = rearrange(patches, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", h=h, p1=patch_size, p2=patch_size)
    
    # Remove padding to restore original dimensions
    return padded_img[:, :, :img_size, :img_size]


# def patch_to_img(img, patch_size, img_size = 512):
#     hw = img.shape[1]

#     return rearrange(img, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", h = img_size // patch_size, p1 = patch_size, p2 = patch_size)

def add_noise(img, sigma=0.05):
    # If the input is a NumPy array, convert it to a PyTorch tensor
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).float()  # Ensure the tensor is float type for adding noise
    
    noise = torch.randn(img.shape) * sigma
    return img + noise.to(img.device)

def to_numpy(img):
    if img.dim() == 3:
        img = img.unsqueeze(0)
    elif img.dim() == 4 and img.size(1) == 1:
        img = img.squeeze(1)

    img = img.permute(0, 2, 3, 1)

    return img.detach().cpu().numpy()

def to_tensor(numpy_img):
    img = torch.from_numpy(numpy_img)

    if img.dim() == 4 and img.size(1) == 1:
        img = img.squeeze(1)
    elif img.dim() == 3:
        img = img.unsqueeze(0)

    img = img.permute(0, 3, 1, 2)

    return img

def get_sample_img(img_size=256):
    image_path = "../datasets/DIV2K_train_HR/0040.png"
    img = Image.open(image_path)

    # Crop to square based on the smaller dimension
    min_dim = min(img.width, img.height)
    left = (img.width - min_dim) / 2
    top = (img.height - min_dim) / 2
    right = (img.width + min_dim) / 2
    bottom = (img.height + min_dim) / 2
    img_cropped = img.crop((left, top, right, bottom))

    # Resize to the desired img_size
    img_resized = img_cropped.resize((img_size, img_size))

    return np.array(img_resized).astype(float) / 255

def clamp(img):
    img[img<0] = 0
    img[img>1] = 1
    
    return img
# def get_sample_img(img_size=256):
#     image_path = "../datasets/DIV2K_train_HR/0032.png"
#     img = Image.open(image_path)
    
#     if img.width > img.height: 
#         img_resized = img.resize((img_size, int(img_size * img.height / img.width))) 
#     else:
#        img_resized = img.resize((int(img_size * img.width / img.height), img_size))

#     # Crop the image to 256x256 from the center
#     width, height = img_resized.size
#     left = (width - img_size) / 2
#     top = (height - img_size) / 2
#     right = (width + img_size) / 2
#     bottom = (height + img_size) / 2
#     img_cropped = img_resized.crop((left, top, right, bottom))
    
#     return np.array(img_cropped).astype(float) / 255