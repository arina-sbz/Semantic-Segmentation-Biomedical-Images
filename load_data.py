import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class WarwickDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment=False):
        """
        Custom Dataset for loading images and masks from the Warwick dataset

        Parameters:
            root_dir: Directory with all the images and masks
            transform: Dictionary containing 'image' and 'mask' transformations
            augment: If True, apply data augmentations
        """
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
        self.image_files = sorted([os.path.join(root_dir, f)
                                  for f in os.listdir(root_dir) if f.startswith('image')])
        self.mask_files = sorted([os.path.join(root_dir, f)
                                 for f in os.listdir(root_dir) if f.startswith('label')])

    def __len__(self):
        """
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the given index

        Parameters:
            idx: Index of the sample to retrieve

        Returns:
            tuple: (image, mask) where image is the transformed input image and
                   mask is the corresponding transformed mask.
        """
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        if self.transform:
            if self.augment:  # Apply augmentations only if augment is True
                image, mask = self.transform['augment'](image, mask)
            image = self.transform['image'](image)
            mask = self.transform['mask'](mask)

        return image, mask


class RandomTransform:
    def __call__(self, image, mask):
        """
        Apply transformations to both the image and mask.
        Parameters:
            image: Input image
            mask: Input mask
        
        Returns:
            Transformed image and mask
        """
        
        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Random rotation
        angle = random.randint(-30, 30)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)

        return image, mask
    
default_transforms = {
    'augment': RandomTransform(), # Apply random transformations
    'image': transforms.Compose([
        transforms.ToTensor(), # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize the image
    ]),
    'mask': transforms.Compose([
        transforms.ToTensor() # Convert mask to tensor
    ])
}
