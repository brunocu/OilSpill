import os
import rasterio
import argparse

def get_image_shape(file_path):
    with rasterio.open(file_path) as dataset:
        return dataset.width, dataset.height

def verify_images_shape(directory):
    shapes = set()
    for filename in os.listdir(directory):
        if filename.endswith(".tif"):  # Assuming SAR images are in .tif format
            file_path = os.path.join(directory, filename)
            shape = get_image_shape(file_path)
            shapes.add(shape)
    
    if len(shapes) == 1:
        print("All images have the same shape.", shapes)
    else:
        print("Images have different shapes:", shapes)
        max_side_length = max(max(shape) for shape in shapes)
        print("Maximum side length among images:", max_side_length)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify image shapes in a directory.")
    parser.add_argument("directory", type=str, help="Directory containing .tif images")
    args = parser.parse_args()
    
    directory = args.directory
    verify_images_shape(directory)
