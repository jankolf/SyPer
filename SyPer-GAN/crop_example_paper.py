from pathlib import Path
from typing import List

import cv2
import torch
import torch.nn.functional as F
import torchvision.utils as torch_utils

from tqdm import tqdm

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import argparse

from align_trans import norm_crop, arcface_ref_points

from facenet_pytorch import MTCNN

mtcnn = MTCNN(
        select_largest=True, min_face_size=50, image_size=150, post_process=False, device="cuda:0"
    )



from mtcnn_periocular_crop import circle_landmarks, transform_axis, normalize_image, norm_crop



if __name__ == "__main__":
    
    image_names : List[Path] = [sorted(Path("./FFHQ/images1024x1024/").resolve().glob("*.png"))[100]]
    
    output_path = "/home/jkolf/projects/GAN/FFHQ/ffhq_cropped/"
    
    for path in tqdm(image_names):
        image_path = str(path)
        image_name = path.name
        
        L_name = image_name.replace(".png", "_left.jpg")
        R_name = image_name.replace(".png", "_right.jpg")
    
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image)
        
        img = transform_axis(image)
        img = normalize_image(image)
        
        img_small = transform_axis(image)
        img_small = normalize_image(image)
        
        boxes, probs, landmarks = mtcnn.detect(img_small, landmarks=True)

        if landmarks is None:
            print("HECZ")
        
        facial5points = landmarks[0]
        f5p_big = (facial5points/img_small.shape[0])*img.shape[0]

        image_big = img.cpu().numpy().astype(np.uint8).copy()
        
        cv2.imwrite(f"alignment_vis_step1_image.jpg", cv2.cvtColor(image_big, cv2.COLOR_RGB2BGR))
        
        image_landmarks = image_big.copy()
        for i in f5p_big:
            img_rectl = cv2.circle(image_landmarks, (int(i[0]),int(i[1])), 8, (69,181,170), 16)
        
        cv2.imwrite(f"alignment_vis_step2_landmarks.jpg", cv2.cvtColor(img_rectl, cv2.COLOR_RGB2BGR))
        
        image_size = 1000
        image_big, projected_landmarks = norm_crop(image_big, f5p_big, image_size=image_size)
        
        
    
        
        image_big = cv2.copyMakeBorder(image_big, 250, 250, 250, 250, cv2.BORDER_REPLICATE)
        projected_landmarks = projected_landmarks + 250
        
        cv2.imwrite(f"alignment_vis_step3_alignmend.jpg", cv2.cvtColor(image_big, cv2.COLOR_RGB2BGR))
        
        eye_l = projected_landmarks[0].astype(int)
        eye_r = projected_landmarks[1].astype(int)
        
        dist = np.abs(eye_l[0] - eye_r[0]) // 2
        
        img_per_region = cv2.rectangle(image_big.copy(), (eye_l[0]-dist, eye_l[1]-dist),(eye_l[0]+dist, eye_l[1]+dist), (221,65,36), 15)
        img_per_region = cv2.rectangle(img_per_region, (eye_r[0]-dist, eye_r[1]-dist),(eye_r[0]+dist, eye_r[1]+dist), (221,65,36), 15)
        img_per_region = cv2.circle(img_per_region, (eye_l[0], eye_l[1]), 10, (69,181,170), 18)
        img_per_region = cv2.circle(img_per_region, (eye_r[0], eye_r[1]), 10, (69,181,170), 18)
        
        cv2.imwrite(f"alignment_vis_step4_region.jpg", cv2.cvtColor(img_per_region, cv2.COLOR_RGB2BGR))
        
        periocular_left     = image_big[eye_l[1]-dist:eye_l[1]+dist, eye_l[0]-dist:eye_l[0]+dist]
        periocular_right    = image_big[eye_r[1]-dist:eye_r[1]+dist, eye_r[0]-dist:eye_r[0]+dist]
        
        periocular_left     = cv2.cvtColor(periocular_left, cv2.COLOR_RGB2BGR)
        periocular_right    = cv2.cvtColor(periocular_right, cv2.COLOR_RGB2BGR)
        
        periocular_left     = cv2.resize(periocular_left,  (224, 224), interpolation=cv2.INTER_CUBIC)
        periocular_right    = cv2.resize(periocular_right, (224, 224), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(f"alignment_vis_step5_crop_left.jpg", periocular_left)
        cv2.imwrite(f"alignment_vis_step5_crop_right.jpg", periocular_right)