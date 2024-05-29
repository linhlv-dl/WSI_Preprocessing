import os
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torchvision
import numpy as np
from tqdm import tqdm
import random
import re
import glob
import itertools
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import cv2


from pytorch_lightning import LightningDataModule

"""
    Class: Tile_Dataset
    Create a Tile dataset
"""
class Tile_Dataset(Dataset):
    """
        Function: __init__
        The initialization function.
    """
    def __init__(self, list_images, transf=None):
        self.list_images = list_images
        self.transf = transf

    """
        Function: __len__
        Return the size of the dataset.
    """
    def __len__(self):
        return len(self.list_images)
    """
        Function: __getitem__
        Get the tile by using id.

        Parameters:
            - idx: The id of the tile

        Returns:
            - The color image array at idx.
            - The gray image array at idx.
    """
    def __getitem__(self, idx):
        image_rgb = self.list_images[idx]
        image_rgb = np.asarray(image_rgb)
        image_gray = cv2.cvtColor(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)

        if self.transf is not None:
            transformed = self.transf(image = image_rgb, image0 = image_gray)
            color_image = transformed['image']
            gray_image = transformed['image0']

        return color_image, gray_image

"""
    Function: get_data_whole_image
    Read the data and create the dataloaders.
    
    Parameters:
        - root_folder: The root folder contains the tiles.
        - train_files: The list of train files.
        - valid_files: The list of validation files.
        - batch_size: The batch size to create the dataloader.

    Returns:
        The training and validation dataloaders.
"""
def get_data_whole_image(root_folder, train_files, valid_files, batch_size):
    train_files = [os.path.join(root_folder, ft) for ft in train_files]
    valid_files = [os.path.join(root_folder, fv) for fv in valid_files]
    #img_dir = os.path.join(root_dir, image_folder)
    #mask_dir = os.path.join(root_dir, mask_folder)
    #train_files = train_files[:1]
    #valid_files = train_files[:1]

    train_images = [] # list of images
    for tf in train_files:
        npz_image = np.load(tf)['arr_0']
        print(npz_image.shape)
        for img in npz_image:
            train_images.append(Image.fromarray(img.astype(np.uint8)).convert('RGB'))

    train_images = random.choices(train_images, k = 60000)

    train_transf = A.Compose([A.CenterCrop(height = 224, width = 224),
                                A.VerticalFlip(p=0.5),
                                A.HorizontalFlip(p=0.5),
                                A.Normalize(mean = [0.5, 0.5, 0.5], std =[0.5, 0.5, 0.5], max_pixel_value = 255.0,),
                                ToTensorV2(),],
                                additional_targets = {'image0': 'image'})

    train_dataset = Tile_Dataset(train_images, train_transf)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=10)

    # Validation
    # valid_images = [] # list of images
    # for vf in valid_files:
    #     npz_vimage = np.load(vf)['arr_0']
    #     print(npz_vimage.shape)
    #     for vimg in npz_vimage:
    #         valid_images.append(Image.fromarray(vimg.astype(np.uint8)).convert('RGB'))

    # valid_transf = A.Compose([A.CenterCrop(height = 224, width = 224),
    #                             A.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5], max_pixel_value = 255.0,),
    #                             ToTensorV2(),],
    #                             additional_targets = {'image0': 'image'})
    
    # valid_dataset = Tile_Dataset(valid_images, valid_transf)
    # valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=10)
    valid_loader = None

    print('Total training images {}'.format(train_dataset.__len__()))
    #print('Total validation images {}'.format(valid_dataset.__len__()))
    
    return train_loader, valid_loader
