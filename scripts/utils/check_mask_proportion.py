import os
import rasterio
import argparse
import numpy as np  # for potential use

def get_mask_proportions(file_path):
    # Open the mask file and read the first band
    with rasterio.open(file_path) as dataset:
        mask = dataset.read(1)
    
    white = (mask == 1).sum()
    black = (mask == 0).sum()
    return white, black

def verify_masks_proportion(directory):
    total_white = 0
    total_black = 0
    # List and process mask images in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".tif"):
            file_path = os.path.join(directory, filename)
            white, black = get_mask_proportions(file_path)
            total_white += white
            total_black += black
    
    # New aggregated calculation for entire dataset
    grand_total = total_white + total_black
    grand_prop_white = total_white / grand_total
    grand_prop_black = total_black / grand_total
    print(f"White = {grand_prop_white:.2%}, Black = {grand_prop_black:.2%}")
    pos_weight = 2 * total_black / (grand_total + 1e-6)
    neg_weight = 2 * total_white / (grand_total + 1e-6)
    print(f"pos_weight = {pos_weight:.6f}, neg_weight = {neg_weight:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the proportion of white vs black pixels in a mask dataset.")
    parser.add_argument("directory", type=str, help="Directory containing mask .tif images")
    args = parser.parse_args()

    verify_masks_proportion(args.directory)