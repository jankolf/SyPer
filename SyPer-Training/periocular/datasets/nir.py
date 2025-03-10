from enum import Enum
from pathlib import Path
import queue as Queue
import threading
import math
from itertools import combinations

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

import numpy as np

class NIRDataset(Dataset):

    def __init__(self,
                 img_size : int,
                 flip_L_to_R : bool,
                 dataset_root : str,
                 img_name_file : str,
                 clean_labels : bool = True
                 ):
        
        self.img_size = img_size
        self.flip_L_to_R = flip_L_to_R
        self._dataset_root  = dataset_root
        self._location_imgs = Path(dataset_root).resolve()

        self._image_names = []
        self._image_labels = []
        with open(self._location_imgs / img_name_file, "r") as f:
            for line in f.readlines():
                line = line.strip()
                self._image_names.append(line)
                label = int(line.split("D")[0][2:])
                
                self._image_labels.append(label)
                
        if clean_labels:
            mapping = {}
            max_key = 0
            
            for idx in range(len(self._image_labels)):
            
                label = self._image_labels[idx]
                
                if label in mapping.keys():
                    self._image_labels[idx] = mapping[label]
                    continue
                
                mapping[label] = max_key
                self._image_labels[idx] = max_key
                max_key += 1
                
        #self._image_names = [line.strip() for line in f.readlines()]

        self._num_classes = len(np.unique(self._image_labels))
        
        assert self._num_classes == 180

        self.transform = transforms.Compose(
            [
             transforms.Resize((img_size,img_size)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])


    @property
    def image_names(self):
        return self._image_names


    @property
    def num_classes(self):
        return self._num_classes

    
    def __len__(self):
        return len(self._image_names)

    
    def __getitem__(self, item_idx):
        image = self.load_image(
                    self._image_names[item_idx]
                )
        #class_label = int(self._image_names[item_idx].split("D")[0][2:])
        class_label = self._image_labels[item_idx]
        class_label = torch.tensor(class_label, dtype=torch.long)
        return image, class_label


    def load_image(self, img_path):
        img = Image.open(self._location_imgs / img_path)
        img = self.transform(img)

        if self.flip_L_to_R and "_left" in img_path:
            img = TF.hflip(img)

        return img



class NIRTrain(NIRDataset):

    def __init__(self,
                 img_size : int = 224,
                 flip_L_to_R : bool = True,
                 dataset_root : str = "/data/jkolf/PeriocularExtension/data/CASIA-Cropped"
                 ):

        super(NIRTrain, self).__init__(
                img_size=img_size,
                flip_L_to_R=flip_L_to_R,
                dataset_root=dataset_root,
                img_name_file="training_images.txt"
        )


class NIRTest(NIRDataset):

    def __init__(self,
                 img_size : int = 224,
                 flip_L_to_R : bool = True,
                 dataset_root : str = "/data/jkolf/PeriocularExtension/data/CASIA-Cropped"
                 ):

        super(NIRTest, self).__init__(
                img_size=img_size,
                flip_L_to_R=flip_L_to_R,
                dataset_root=dataset_root,
                img_name_file="test_images.txt"
        )
        
        self._test_combinations = []
        for (L,R) in combinations(self._image_names, 2):
            ID_L = L.strip()[2:6]
            ID_R = R.strip()[2:6]
            
            if ID_L != ID_R:
                self._test_combinations.append((L,R,0))
                continue
            
            if "left" in L and "left" in R:
                self._test_combinations.append((L,R,1))
                continue
                
            if "right" in L and "right" in R:
                self._test_combinations.append((L,R,1))
                continue
            
        self._image_labels = torch.tensor(self._image_labels)


    def __len__(self):
        return len(self._test_combinations)


    def __getitem__(self, item_idx):
        return self._test_combinations[item_idx]