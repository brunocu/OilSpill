import os
import rasterio
import argparse

def get_image_shape(file_path):
    with rasterio.open(file_path) as dataset:
        return dataset.width, dataset.height

def verify_images_shape(directory, expected_shape=(2048, 2048)):
    invalid_images = []
    shapes = set()
    for filename in os.listdir(directory):
        if filename.endswith(".tif"):  # Assuming SAR images are in .tif format
            file_path = os.path.join(directory, filename)
            shape = get_image_shape(file_path)
            shapes.add(shape)
            if shape != expected_shape:
                invalid_images.append((filename, shape))
    
    if len(invalid_images) == 0:
        print("All images have the expected shape:", expected_shape)
    else:
        print("Images that do not have shape", expected_shape, ":")
        for filename, shape in invalid_images:
            print(f"{filename}: {shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify image shapes in a directory.")
    parser.add_argument("directory", type=str, help="Directory containing .tif images")
    args = parser.parse_args()
    
    directory = args.directory
    verify_images_shape(directory)
