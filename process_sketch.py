import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from annotator.sketch import SketchDetector

def process_images_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    sketch_detector = SketchDetector()

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)

            if img is not None:
                sketch_image = sketch_detector(img)
                
                sketch_image_path = os.path.join(output_folder, filename)
                cv2.imwrite(sketch_image_path, sketch_image)
                print(f"Processed and saved sketch image: {sketch_image_path}")
            else:
                print(f"Failed to read image: {img_path}")

input_folder = "path/to/images"
output_folder = "path/to/sketches"

process_images_in_folder(input_folder, output_folder)
