import os
import shutil
import csv

# Define paths
original_dataset_path = 'data/raw'
output_dataset_path = 'data/test'
images_source_path = os.path.join(original_dataset_path, 'Images')
masks_source_path = os.path.join(original_dataset_path, 'Mask')
images_output_path = os.path.join(output_dataset_path, 'images')
labels_output_path = os.path.join(output_dataset_path, 'mask')
csv_filepath = os.path.join(output_dataset_path, 'records.csv')

# Ensure output directories exist
os.makedirs(images_output_path, exist_ok=True)
os.makedirs(labels_output_path, exist_ok=True)

# Prepare CSV recording
records = []

# List all label folders in the images source directory
labels = [d for d in os.listdir(images_source_path) if os.path.isdir(os.path.join(images_source_path, d))]

for label in labels:
    label_lower = label.lower()
    image_dir_path = os.path.join(images_source_path, label)
    mask_dir_path = os.path.join(masks_source_path, label)
    # List all .tif image files
    image_files = [f for f in os.listdir(image_dir_path)
                   if os.path.isfile(os.path.join(image_dir_path, f)) and f.lower().endswith('.tif')]
    for image_file in image_files:
        name, ext = os.path.splitext(image_file)
        mask_file = f"{name}_segmentation{ext}"
        src_image_path = os.path.join(image_dir_path, image_file)
        src_mask_path = os.path.join(mask_dir_path, mask_file)
        new_image_name = f"{label_lower}_{image_file}"
        new_mask_name = f"{label_lower}_{mask_file}"
        dst_image_path = os.path.join(images_output_path, new_image_name)
        dst_mask_path = os.path.join(labels_output_path, new_mask_name)
        
        # Verify both image and mask files exist before moving
        if not os.path.exists(src_image_path):
            print(f"Image file does not exist: {src_image_path}")
            continue
        if not os.path.exists(src_mask_path):
            print(f"Mask file does not exist: {src_mask_path}")
            continue
        
        # Move image and mask
        shutil.move(src_image_path, dst_image_path)
        shutil.move(src_mask_path, dst_mask_path)
        
        # Record the target paths and label
        records.append([label, dst_image_path, dst_mask_path])
    
    # Delete origin directories if they're empty after moving
    if not os.listdir(image_dir_path):
        os.rmdir(image_dir_path)
    if not os.listdir(mask_dir_path):
        os.rmdir(mask_dir_path)

# Write records to csv file
with open(csv_filepath, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Label', 'Image', 'Mask'])
    writer.writerows(records)

print("Restructuring complete and records saved to csv!")
