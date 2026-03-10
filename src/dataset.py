# -*- coding: UTF-8 -*-
# @Time    : 2024/06/07  
# @Author  : Gemini
# @File    : loaders_radio_lightm_unet.py
# @Description: Data loader for Radio-MambaNet v14.
#              Same data loading logic as v12, compatible with simplified v14 configuration.

from __future__ import print_function, division
import os
import warnings
import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms


import numpy as np
import os
import re
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

warnings.filterwarnings("ignore")


class LocationDataset(Dataset):
    """
    Location Dataset Loader for Radio-MambaNet.
    Input: Building map (1 channel), Transmitter location map (1 channel), and buildings channel (1 channel).
    Output: Pathloss map (1 channel).
    Note: Buildings and Tx images are 256x256, need to be resized to match pathloss resolution (512x512 or 1024x1024).
    """

    def __init__(self, maps_inds=None, phase="train",
                 ind1=0, ind2=0,
                 thresh=0.0,
                 resolution=512,  # 512 or 1024
                 dir_pathloss=None,  # If None, will auto-select based on resolution
                 dir_buildings="/home/zqm1/dataset/Location/cond/buildings_complete",
                 dir_tx="/home/zqm1/dataset/Location/cond/xy_pngs",
                 transform=transforms.ToTensor()):
        """
        Args:
            maps_inds (np.ndarray, optional): Shuffled building indices.
            phase (str): "train", "val", or "test".
            ind1, ind2 (int): Start and end indices for custom data splits.
            thresh (float): Pathloss threshold for normalization.
            resolution (int): Output resolution, 512 or 1024. Default is 512.
            dir_pathloss (str, optional): Directory containing pathloss images. 
                If None, will auto-select: location_512/pathloss_process for 512, location_1024/pathloss_process for 1024.
            dir_buildings (str): Directory containing building images (256x256).
            dir_tx (str): Directory containing transmitter location images (256x256).
            transform (callable, optional): Transform to be applied to the images.
        """
        assert resolution in [512, 1024], f"resolution must be 512 or 1024, got {resolution}"
        self.resolution = resolution
        self.thresh = thresh
        
        # Auto-select pathloss directory if not provided
        if dir_pathloss is None:
            if resolution == 512:
                dir_pathloss = "/home/zqm1/dataset/Location/location_512/pathloss_process"
            else:  # 1024
                dir_pathloss = "/home/zqm1/dataset/Location/location_1024/pathloss_process"
        
        self.dir_pathloss = dir_pathloss
        self.dir_buildings = dir_buildings
        self.dir_tx = dir_tx
        self.transform = transform
        
        # Get all pathloss files and parse them
        pathloss_files = sorted([f for f in os.listdir(dir_pathloss) if f.endswith('.png')])
        
        # Parse filenames to extract building, x, y
        # Format: {building}_X{x}_Y{y}.png, e.g., 0_X102_Y65.png
        all_samples = []
        unique_buildings = set()
        for filename in pathloss_files:
            base_name = filename.replace('.png', '')
            parts = base_name.split('_')
            # Expected format: ['0', 'X102', 'Y65'] -> 3 parts
            if len(parts) == 3:
                try:
                    building = int(parts[0])
                    # Extract x from 'X102' -> '102'
                    if parts[1].startswith('X'):
                        x = int(parts[1][1:])  # Remove 'X' prefix
                    else:
                        continue
                    # Extract y from 'Y65' -> '65'
                    if parts[2].startswith('Y'):
                        y = int(parts[2][1:])  # Remove 'Y' prefix
                    else:
                        continue
                    unique_buildings.add(building)
                    all_samples.append({
                        'building': building,
                        'x': x,
                        'y': y,
                        'pathloss_file': filename
                    })
                except (ValueError, IndexError):
                    # Skip files that don't match the expected format
                    continue
        
        # Create and shuffle building indices (similar to RadioMambaNetDataset)
        unique_buildings_list = sorted(list(unique_buildings))
        num_buildings = len(unique_buildings_list)
        
        if maps_inds is None:
            self.maps_inds = np.arange(0, num_buildings, 1, dtype=np.int16)
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds = maps_inds
        
        # Split data based on phase (similar to RadioMambaNetDataset)
        if phase == "train":
            self.ind1 = 0
            self.ind2 = int(num_buildings * 0.35) - 1
        elif phase == "val":
            self.ind1 = int(num_buildings * 0.35)
            self.ind2 = int(num_buildings * 0.45) - 1
        elif phase == "test":
            self.ind1 = int(num_buildings * 0.45)
            self.ind2 = int(num_buildings * 0.55) - 1
        # if phase == "test":
        #     self.ind1 = 0
        #     self.ind2 = int(num_buildings * 0.35) - 1
        else:  # custom
            self.ind1 = ind1
            self.ind2 = ind2
        
        # Get selected building IDs based on shuffled indices
        selected_building_ids = set()
        for i in range(self.ind1, self.ind2 + 1):
            building_idx = self.maps_inds[i]
            selected_building_ids.add(unique_buildings_list[building_idx])
        
        # Filter samples to only include selected buildings
        self.samples = [s for s in all_samples if s['building'] in selected_building_ids]
        
        self.height = resolution  # Pathloss images resolution
        self.width = resolution

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        building = sample['building']
        x = sample['x']
        y = sample['y']
        pathloss_file = sample['pathloss_file']
        
        # Load pathloss image (512x512 or 1024x1024)
        img_path_pathloss = os.path.join(self.dir_pathloss, pathloss_file)
        image_pathloss = np.asarray(
            io.imread(img_path_pathloss), dtype=np.float32)
        if image_pathloss.ndim == 2:
            image_pathloss = np.expand_dims(image_pathloss, axis=2)
        image_pathloss = image_pathloss / 255.0
        
        # Resize pathloss to target resolution if needed
        if image_pathloss.shape[0] != self.resolution or image_pathloss.shape[1] != self.resolution:
            from PIL import Image
            image_pathloss_pil = Image.fromarray((image_pathloss.squeeze() * 255).astype(np.uint8))
            image_pathloss_pil = image_pathloss_pil.resize((self.resolution, self.resolution), Image.BILINEAR)
            image_pathloss = np.asarray(image_pathloss_pil, dtype=np.float32) / 255.0
            if image_pathloss.ndim == 2:
                image_pathloss = np.expand_dims(image_pathloss, axis=2)
        
        # Pathloss threshold transform
        if self.thresh > 0:
            mask = image_pathloss < self.thresh
            image_pathloss[mask] = self.thresh
            image_pathloss = image_pathloss - self.thresh * np.ones(np.shape(image_pathloss))
            image_pathloss = image_pathloss / (1 - self.thresh)
        
        # Load building image (256x256) and resize to target resolution
        img_path_buildings = os.path.join(self.dir_buildings, f"{building}.png")
        image_buildings = np.asarray(
            io.imread(img_path_buildings), dtype=np.float32) / 255.0
        if image_buildings.ndim == 2:
            image_buildings = np.expand_dims(image_buildings, axis=2)
        
        # Resize building image from 256x256 to target resolution
        from PIL import Image
        image_buildings_pil = Image.fromarray((image_buildings.squeeze() * 255).astype(np.uint8))
        image_buildings_pil = image_buildings_pil.resize((self.resolution, self.resolution), Image.BILINEAR)
        image_buildings = np.asarray(image_buildings_pil, dtype=np.float32) / 255.0
        if image_buildings.ndim == 2:
            image_buildings = np.expand_dims(image_buildings, axis=2)
        
        # Load transmitter location image (256x256) and resize to target resolution
        img_path_tx = os.path.join(self.dir_tx, f"{building}_{x}_{y}.png")
        image_tx = np.asarray(io.imread(img_path_tx), dtype=np.float32) / 255.0
        if image_tx.ndim == 2:
            image_tx = np.expand_dims(image_tx, axis=2)
        
        # Resize Tx image from 256x256 to target resolution
        image_tx_pil = Image.fromarray((image_tx.squeeze() * 255).astype(np.uint8))
        image_tx_pil = image_tx_pil.resize((self.resolution, self.resolution), Image.BILINEAR)
        image_tx = np.asarray(image_tx_pil, dtype=np.float32) / 255.0
        if image_tx.ndim == 2:
            image_tx = np.expand_dims(image_tx, axis=2)
        
        # Create third channel: randomly sample 1% of pixels from image_pathloss
        h, w = image_pathloss.shape[:2]
        total_pixels = h * w
        num_samples = int(total_pixels * 0.01)  # 1% sampling rate
        
        # Create a zero matrix with same shape as image_pathloss
        third_channel = np.zeros_like(image_pathloss)
        
        # Randomly select pixel positions
        np.random.seed(None)  # Use different seed for each sample
        flat_indices = np.random.choice(total_pixels, size=num_samples, replace=False)
        row_indices = flat_indices // w
        col_indices = flat_indices % w
        
        # Copy sampled values from image_pathloss
        third_channel[row_indices, col_indices, :] = image_pathloss[row_indices, col_indices, :]
        
        # Concatenate to form a 3-channel input
        # (Buildings, Tx, Sampled Pathloss)
        inputs_numpy = np.concatenate(
            [image_buildings, image_tx, third_channel], axis=2)
        
        if self.transform:
            inputs = self.transform(inputs_numpy).type(torch.float32)
            image_pathloss = self.transform(image_pathloss).type(torch.float32)
        else:  # Fallback if no transform is provided
            inputs = torch.from_numpy(
                inputs_numpy.transpose((2, 0, 1))).type(torch.float32)
            image_pathloss = torch.from_numpy(
                image_pathloss.transpose((2, 0, 1))).type(torch.float32)
        
        # Return with filename for identification
        filename = pathloss_file.replace('.png', '')
        return inputs, image_pathloss, filename



class RadioMambaNetDataset(Dataset):
    """
    RadioMapSeer Data Loader for Radio-MambaNet.
    Updated to support both with/without cars simulation and input.
    Input: Building map (1 channel), Transmitter location map (1 channel), and cars/buildings channel (1 channel).
    Output: DPM path loss map (1 channel).
    """

    def __init__(self, maps_inds=None, phase="train",
                 ind1=0, ind2=0,
                 thresh = 0.0,
                 dir_dataset="RadioMapSeer/",
                 numTx=80,
                 carsSimul="no",
                 carsInput="no",
                 transform=transforms.ToTensor()):
        """
        Args:
            maps_inds (np.ndarray, optional): Shuffled map indices.
            phase (str): "train", "val", or "test".
            ind1, ind2 (int): Start and end indices for custom data splits.
            dir_dataset (str): Root directory of the RadioMapSeer dataset.
            numTx (int): Number of transmitters per map.
            carsSimul (str): "no" or "yes". Use simulation with or without cars. Default="no".
            carsInput (str): "no" or "yes". Take inputs with or without cars channel. Default="no".
            transform (callable, optional): Transform to be applied to the images.
        """

        if maps_inds is None:
            self.maps_inds = np.arange(0, 700, 1, dtype=np.int16)
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds = maps_inds

        if phase == "train":
            self.ind1 = 0
            self.ind2 = 549
        elif phase == "val":
            self.ind1 = 550
            self.ind2 = 599
        elif phase == "test":
            self.ind1 = 600
            self.ind2 = 699
        else:  # custom
            self.ind1 = ind1
            self.ind2 = ind2

        self.thresh = thresh
        self.dir_dataset = dir_dataset
        self.numTx = numTx
        self.carsSimul = carsSimul
        self.carsInput = carsInput

        # Setup gain directory based on cars simulation
        if carsSimul == "no":
            self.dir_gain = os.path.join(self.dir_dataset, "gain", "DPM")
        else:
            self.dir_gain = os.path.join(self.dir_dataset, "gain", "carsDPM")
        
        self.dir_buildings = os.path.join(
            self.dir_dataset, "png", "buildings_complete")
        self.dir_Tx = os.path.join(self.dir_dataset, "png", "antennas")
        
        # Setup cars directory if needed
        if carsInput != "no":
            self.dir_cars = os.path.join(self.dir_dataset, "png", "cars")

        self.transform = transform
        self.height = 256
        self.width = 256

    def __len__(self):
        return (self.ind2 - self.ind1 + 1) * self.numTx

    def __getitem__(self, idx):
        map_idx_in_split = idx // self.numTx
        tx_idx_in_map = idx % self.numTx
        dataset_map_ind = self.maps_inds[self.ind1 + map_idx_in_split] + 1

        name1 = str(dataset_map_ind) + ".png"
        name2 = str(dataset_map_ind) + "_" + str(tx_idx_in_map) + ".png"

        # Load building map
        img_name_buildings = os.path.join(self.dir_buildings, name1)
        image_buildings = np.asarray(
            io.imread(img_name_buildings), dtype=np.float32) / 255.0

        # Load transmitter map
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx), dtype=np.float32) / 255.0

        # Load ground truth gain map
        img_name_gain = os.path.join(self.dir_gain, name2)
        image_gain = np.expand_dims(
            io.imread(img_name_gain).astype(np.float32), axis=2) / 255.0
        
        # pathloss threshold transform
        if self.thresh > 0:
            mask = image_gain < self.thresh
            image_gain[mask] = self.thresh
            image_gain = image_gain-self.thresh*np.ones(np.shape(image_gain))
            image_gain = image_gain/(1-self.thresh)

        # Ensure channel dimension exists
        if image_buildings.ndim == 2:
            image_buildings = np.expand_dims(image_buildings, axis=2)
        if image_Tx.ndim == 2:
            image_Tx = np.expand_dims(image_Tx, axis=2)

        # Prepare third channel based on cars input setting
        if self.carsInput == "no":
            # Use buildings as third channel (same as before)
            third_channel = image_buildings
        else:
            # Load cars map for third channel
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars), dtype=np.float32) / 255.0
            if image_cars.ndim == 2:
                image_cars = np.expand_dims(image_cars, axis=2)
            third_channel = image_cars

        # Concatenate to form a 3-channel input
        # (Buildings, Tx, Buildings/Cars)
        inputs_numpy = np.concatenate(
            [image_buildings, image_Tx, third_channel], axis=2)

        if self.transform:
            inputs = self.transform(inputs_numpy).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
        else:  # Fallback if no transform is provided
            inputs = torch.from_numpy(
                inputs_numpy.transpose((2, 0, 1))).type(torch.float32)
            image_gain = torch.from_numpy(
                image_gain.transpose((2, 0, 1))).type(torch.float32)

        return inputs, image_gain, name2
