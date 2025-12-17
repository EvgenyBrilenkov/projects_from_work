import cv2
import numpy as np
import os

def mask_to_yolo_txt(mask_path, txt_path, class_id=0, simplify_epsilon=1.0):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    h, w = mask.shape

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    with open(txt_path, 'w') as f:
        for contour in contours:
            contour = contour.reshape(-1, 2)
            contour = contour.astype(float)
            contour[:, 0] /= w
            contour[:, 1] /= h
            contour_flat = ' '.join(map(str, contour.flatten()))
            f.write(f"{class_id} {contour_flat}\n")
    
    return mask

mask_dir = "/wrk/main/yolo/data/masks_val"
txt_dir = "/wrk/main/yolo/yolo_data/labels/train"

for mask_file in os.listdir(mask_dir):
    mask_path = os.path.join(mask_dir, mask_file)
    txt_path = os.path.join(txt_dir, mask_file.replace(".png", ".txt"))
    mask_to_yolo_txt(mask_path, txt_path, class_id=0)