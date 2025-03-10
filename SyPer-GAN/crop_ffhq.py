from pathlib import Path
from typing import List

import cv2
import torch
import torch.nn.functional as F
import torchvision.utils as torch_utils

from tqdm import tqdm

from mtcnn_periocular_crop import crop_image, circle_landmarks


if __name__ == "__main__":
    
    image_names : List[Path] = sorted(Path("./FFHQ/images1024x1024/").resolve().glob("*.png"))
    
    output_path = "./ffhq_cropped/"
    
    for path in tqdm(image_names):
        image_path = str(path)
        image_name = path.name
        
        L_name = image_name.replace(".png", "_left.jpg")
        R_name = image_name.replace(".png", "_right.jpg")
    
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image)
        l, r = crop_image(image, image)
    
        if l is not None and r is not None:
            cv2.imwrite(f"{output_path}{L_name}", l)
            cv2.imwrite(f"{output_path}{R_name}", r)
        
    