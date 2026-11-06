import numpy as np
from PIL import Image, ImageDraw
import scipy
import matplotlib.pyplot as plt
from pathlib import Path

X = 1550
Y = 800
W = 826
H = 170

image_path = "C:/Users/profp/repos/ScientificComputing/experiments/Cornu/photos/Cornu Photo's 1.0/0.5kg.JPG"
calibration_factor = 0.05 # pixels per mm
crop_box = (X, Y, X + W, Y + H)

# Load, draw sanity check and crop
original_image = Image.open(image_path)
img_check = original_image
draw = ImageDraw.Draw(img_check)
draw.rectangle(crop_box, outline='red', width=5)
draw.rectangle((X+W/2, Y+H/2, X+W/2 + 10, Y+H/2 + 10), fill='red')
img_check.save(f'check_{Path(image_path).name}')