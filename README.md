# Oil Spill Detection

## Prerequisites

- **Conda:** Ensure you have Miniconda or Anaconda installed.
- **Git:** To clone the repository and initialize submodules.

## Setup

1. **Clone the Repository and Submodules:**

   ```bash
   git clone --recurse-submodules <repository_url>
   cd oilspill
   ```

2. **Create and Activate the Conda Environment:**

   ```bash
   conda env create -f env/environment.yml
   conda activate oilspill
   ```
3. **Install PaddleSeg in Editable Mode:**

    The `PaddleSeg` folder is included as a git submodule but is not automatically installed. To install it manually, run:

    ```bash
    pip install -r PaddleSeg/requirements.txt
    pip install -e PaddleSeg
    ```


## Data Preparation

1. **Download the Dataset:**

   The `data/raw/` folder contains a `files.txt` file with a list of URLs to download the 7z compressed dataset. To download all parts, you can run a script or use a download manager that supports URL lists. For example, using `wget`:

   ```bash
   cd data/raw
   wget -i files.txt
   ```

2. **Extract the 7z Files:**

   Once downloaded, extract the 7z files to the same `data/raw` folder. You can use a tool like [7-Zip](https://www.7-zip.org/) or a command-line equivalent.

3. **Convert to Model-Ready Structure:**

   After extraction, run the Python preprocessing scripts to reorganize the dataset into the required structure for training and evaluation:

   ```bash
   # Restructure training and validation data
   python scripts/preprocess/restructure.py

   # Restructure test data
   python scripts/preprocess/restructure_test.py
   ```

   The processed data will be placed under `data/`, organized into separate folders for training and testing.

## Training

1. **Configure Training:**

   Edit the configuration file `configs/config.yml` as needed to adjust training parameters (dataset paths, model parameters, optimizer settings, etc.).

2. **Start Training:**

   Run the training script:

   ```bash
   bash scripts/train.sh
   ```

   Training outputs (models, logs, etc.) will be saved in the `outputs/` directory.

**Note:** Trained model weights are also stored in the `checkpoints/` folder. To use different weights for inference, update the `--model_path` parameter in the [predict.sh](scripts/predict.sh) script accordingly.

## Inference

1. **Run Predictions:**

   After training, use the prediction script to generate segmentation results:

   ```bash
   bash scripts/predict.sh
   ```

## Additional Tools

- **Visualization Tools:**

  For visualizing training images, annotations, or segmentation predictions, use the viewer script (add the --pred flag for predictions):

  ```bash
  python scripts/viewer.py [--pred]
  ```
