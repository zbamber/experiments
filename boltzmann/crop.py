import os
import glob
from PIL import Image

def batch_crop_images():
    # Set the input directory ('.' means the current directory)
    input_dir = 'data'
    output_dir = 'cropped'

    # Create the output directory if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)

    # Calculate the crop box coordinates
    # PIL's crop uses a tuple of (left, upper, right, lower)
    original_width = 3200
    original_height = 2400
    
    left = 191
    top = 193
    right = original_width - 300  # 2884
    bottom = original_height - 176 - 637 # 2224
    
    crop_box = (left, top, right, bottom)

    # Find all .png files in the input directory
    search_pattern = os.path.join(input_dir, '*.png')
    png_files = glob.glob(search_pattern)

    if not png_files:
        print("No PNG files found in the current directory.")
        return

    print(f"Found {len(png_files)} PNG files. Starting crop...")

    # Process each image
    for file_path in png_files:
        filename = os.path.basename(file_path)
        
        try:
            with Image.open(file_path) as img:
                # Optional: Verify the image is actually 3200x2400 before cropping
                if img.size != (original_width, original_height):
                    print(f"Skipping {filename}: Incorrect size {img.size}")
                    continue

                # Crop the image
                cropped_img = img.crop(crop_box)
                
                # Save it to the 'cropped' directory
                output_path = os.path.join(output_dir, filename)
                cropped_img.save(output_path)
                print(f"Successfully cropped: {filename}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("Batch crop complete!")

if __name__ == "__main__":
    batch_crop_images()