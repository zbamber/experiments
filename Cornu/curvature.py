import numpy as np
from PIL import Image, ImageDraw
import scipy
import matplotlib.pyplot as plt
from pathlib import Path

X = 1550
Y = 800
W = 826
H = 170

image_path = "/photos/Cornu Photo's 1.0/0.5kg.JPG"
calibration_factor = 0.05 # pixels per mm
crop_box = (X, Y, X + W, X + H)

# load, draw sanity check and crop
original_image = Image.open(image_path)
img_check = original_image
draw = ImageDraw.Draw(img_check)
draw.rectangle(crop_box, outline='red', width=5)
img_check.save(f'check_{Path(image_path).name}')
img_cropped = original_image.crop(crop_box).convert('L')
img_array = np.array(img_cropped)

# get the intensity profile averaged across a 10 pixel band
center_y = img_array.shape[0] // 2
intensity_profile = np.mean(img_array[center_y - 5 : center_y + 5, :], axis=0)

# find dark fringes (peaks of the inverse profile)
inverted_profile = -intensity_profile
peaks, properties = scipy.signal.find_peaks(inverted_profile, distance=15, prominence=5)

# get centre coordinate
center_x = img_array.shape[1] / 2

left_peaks = peaks[peaks < center_x][::-1] # reverse so central peaks are first
right_peaks = peaks[peaks > center_x]

# calculate 2x_n and x_n^2
x_n_squared_values = []
n_values = []

# pair up corresponding fringes
min_length = min(len(left_peaks), len(right_peaks))
for n in range(min_length):
    pixel_distance_2xn = right_peaks[n] - left_peaks[n]
    physical_distance_2xn = pixel_distance_2xn * calibration_factor
    
    x_n = physical_distance_2xn / 2.0
    x_n_squared = x_n ** 2
    
    # assuming the first clear dark fringe pair is n=1
    # adjust n starting value if central spot is a different order
    fringe_order = n + 1 
    
    x_n_squared_values.append(x_n_squared)
    n_values.append(fringe_order)

# find curvature
# equation: n * (lambda / 2) = t_0 + 0.5 * kappa * x_n^2
# so n = (kappa / lambda) * x_n^2 + (2 * t_0 / lambda)
# G = kappa / lambda
lam = 589e-6  # wavelength in mm (e.g., Sodium D lines)

slope, intercept, r_value, p_value, std_err = scipy.signal.linregress(x_n_squared_values, n_values)
kappa = slope * lam

print(f"Calculated curvature (kappa): {kappa:.6f} mm^-1")
print(f"R-squared of linear fit: {r_value**2:.4f}")

# create plot for each to verify
plt.plot(intensity_profile)
plt.plot(peaks, intensity_profile[peaks], "x", color="red")
plt.title("Intensity Profile with Detected Dark Fringes")
plt.xlabel("Distance (pixels)")
plt.ylabel("Intensity")
plt.show()