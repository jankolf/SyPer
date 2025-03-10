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


def normalize_image(img):
    img = img - img.min()
    img = img / (img.max() - img.min())
    img = img * 255.0
    return img


def transform_axis(img):
    if len(img.shape) == 4:
        img = img[0]
        
    if img.shape[0] == 3:
        img = img.permute(1, 2, 0)
        
    return img


def add_circles(img, points, color=(255,0,0), radius=5):
    
    if len(points.shape) == 1:
        points = [points]
        
    for p in points:
        loc = (int(p[0]), int(p[1]))
        img = cv2.circle(img, loc, 3, color, radius)
        
    return img


def circle_landmarks(img):
    
    img_tensor = torch.from_numpy(img)
    img_tensor = transform_axis(img_tensor)
    img_tensor = normalize_image(img_tensor)

    boxes, probs, landmarks = mtcnn.detect(img_tensor, landmarks=True)

    if landmarks is None:
        return None
    
    facial5points = landmarks[0].astype(int)
    
    for l in facial5points:
        img = add_circles(img, l)
    
    im = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img
    

def crop_image(img : torch.Tensor, img_small : torch.Tensor):
    
    img = transform_axis(img)
    img = normalize_image(img)
    
    img_small = transform_axis(img_small)
    img_small = normalize_image(img_small)
    
    boxes, probs, landmarks = mtcnn.detect(img_small, landmarks=True)

    if landmarks is None:
        return None, None
    
    facial5points = landmarks[0]
    f5p_big = (facial5points/img_small.shape[0])*img.shape[0]

    image_big = img.cpu().numpy().astype(np.uint8).copy()
    
    image_size = 1000
    image_big, projected_landmarks = norm_crop(image_big, f5p_big, image_size=image_size)
    
    image_big = cv2.copyMakeBorder(image_big, 250, 250, 250, 250, cv2.BORDER_REPLICATE)
    projected_landmarks = projected_landmarks + 250
    
    eye_l = projected_landmarks[0].astype(int)
    eye_r = projected_landmarks[1].astype(int)
    
    dist = np.abs(eye_l[0] - eye_r[0]) // 2
    
    periocular_left     = image_big[eye_l[1]-dist:eye_l[1]+dist, eye_l[0]-dist:eye_l[0]+dist]
    periocular_right    = image_big[eye_r[1]-dist:eye_r[1]+dist, eye_r[0]-dist:eye_r[0]+dist]
    
    periocular_left     = cv2.cvtColor(periocular_left, cv2.COLOR_RGB2BGR)
    periocular_right    = cv2.cvtColor(periocular_right, cv2.COLOR_RGB2BGR)
    
    periocular_left     = cv2.resize(periocular_left,  (224, 224), interpolation=cv2.INTER_CUBIC)
    periocular_right    = cv2.resize(periocular_right, (224, 224), interpolation=cv2.INTER_CUBIC)
    
    return periocular_left, periocular_right