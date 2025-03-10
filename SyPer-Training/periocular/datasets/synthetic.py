from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image


class NoIDSyntheticDataset(Dataset):
    
    def __init__(   self,
                    img_size : int = 224,
                    flip_L_to_R : bool = True,
                    dataset_root : str = "./data/SyntheticPeriocular"):
    
        super(NoIDSyntheticDataset, self).__init__()
        
        self.transform = transforms.Compose(
            [
             transforms.Resize((img_size,img_size)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        self.flip_L_to_R = flip_L_to_R

        self._dataset_root  = dataset_root
        
        self._image_names   = sorted(Path(dataset_root).resolve().rglob("*.jpg"))
        self._num_images    = len(self._image_names)
        self._dummy_label   = torch.tensor(0, dtype=torch.long)
        
    
    def __len__(self) -> int:
        return self._num_images
    
    
    def __getitem__(self, index) -> torch.Tensor:
        
        path = str(self._image_names[index])
        img = Image.open(path)
        img = self.transform(img)
        
        if self.flip_L_to_R and ("L" in path or "left" in path):
            img = TF.hflip(img)
            
        return img, self._dummy_label