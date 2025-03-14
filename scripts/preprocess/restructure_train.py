import os
import shutil

# Define paths
original_dataset_path = 'data/raw'  # Path to the original dataset directory
output_dataset_path = 'data/train'  # Path to the output dataset directory
images_output_path = os.path.join(output_dataset_path, 'images')
labels_output_path = os.path.join(output_dataset_path, 'mask')

# Ensure output directories exist
os.makedirs(images_output_path, exist_ok=True)
os.makedirs(labels_output_path, exist_ok=True)

# List all subdirectories under the original dataset path
subdirs = [d for d in os.listdir(original_dataset_path)
           if os.path.isdir(os.path.join(original_dataset_path, d))]

for subdir in subdirs:
    # Check if it's a mask directory
    if subdir.lower().startswith('mask_'):
        # Derive the tag from the mask directory
        tag = subdir[5:].lower()  # e.g. 'mask_oil' -> 'oil'
        mask_dir_path = os.path.join(original_dataset_path, subdir)

        # List all mask images, only .tif files
        mask_files = [f for f in os.listdir(mask_dir_path)
                      if os.path.isfile(os.path.join(mask_dir_path, f)) and f.lower().endswith('.tif')]

        # For each mask image, move it to the labels folder
        for mask_file in mask_files:
            new_mask_name = f"{tag}_{mask_file}"
            src_mask_path = os.path.join(mask_dir_path, mask_file)
            dst_mask_path = os.path.join(labels_output_path, new_mask_name)
            shutil.move(src_mask_path, dst_mask_path)
        # Clean up mask directory if empty
        if not os.listdir(mask_dir_path):
            os.rmdir(mask_dir_path)
    else:
        # Otherwise, it should be a directory with actual images
        tag = subdir.lower()  # e.g. 'oil', 'lookalike'
        images_dir_path = os.path.join(original_dataset_path, subdir)

        # List all image files, only .tif files
        image_files = [f for f in os.listdir(images_dir_path)
                       if os.path.isfile(os.path.join(images_dir_path, f)) and f.lower().endswith('.tif')]

        # For each image file, move it to the images folder
        for image_file in image_files:
            new_image_name = f"{tag}_{image_file}"
            src_image_path = os.path.join(images_dir_path, image_file)
            dst_image_path = os.path.join(images_output_path, new_image_name)
            shutil.move(src_image_path, dst_image_path)
        # Clean up image directory if empty
        if not os.listdir(images_dir_path):
            os.rmdir(images_dir_path)

print("Restructuring complete!")
