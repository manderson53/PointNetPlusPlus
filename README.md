# PointNetPlusPlus
A PointNet Plus Plus implementation utilizing the Paris Lille Dataset: https://npm3d.fr/paris-lille-3d

## Overview
This repo contains scripts to preprocess PLY point cloud data for training and validation of PointNet++ models.

## Configuration
- All file paths and output directories are defined in `config_local.py`.
- A template is provided in `config_template.py`.

## Setup

### 1. Clone the repository
git clone https://github.com/manderson53/PointNetPlusPlus/tree/main

### 2. Download datset
Download the Paris-Lille-3D dataset: https://npm3d.fr/paris-lille-3d

### 3. Configure paths
Copy the template configuration and fill in paths.
Update paths in config_local.py to your local directories for training/validation data.

### 4. Create and activate a virtual environment
Windows:
python -m venv venv
venv\Scripts\activate

Linux/macOS:
python3 -m venv venv
source venv/bin/activate

### 5. Install dependencies
pip install -r requirements.txt

## Usage

### 1. Preprocess training and validation data
python ./preprocess_dataset.py
python ./preprocess_validation_dataset.py
python ./compute_xyz_min_max.py  # Only needed if using physics-enhanced loss

### 2. Train the model
python ./PointNet2Training.py
In the script you can set use_physics_loss to true or false to utilize the physics enhanced loss function or not

### 3. Evaluate results:
python ./evaluate.py
In the script you can set use_physics_loss to true or false to evalulate the physics enhanced results or the baseline results

## Notes & Tips
- Preprocessed data and label weights are saved in directories specified in the config.
- The file test.py is only used to test if the model is working do not evaluate results.
- If adding new PLY files, update TRAINING_PLY_FILES or VALIDATION_PLY_FILES in config_local.py and re-run preprocessing.
- Keep your virtual environment up-to-date:
    pip install --upgrade -r requirements.txt

## Next Steps & Future Work
- Extend preprocessing for additional data channels (reflectance is available in the Paris Lille dataset).
- Experiment with different PointNet++ architectures or loss functions.
- Compare standard vs physics-enhanced loss performance.

## References
- Utilized the pointnet++ model from https://github.com/yanx27/Pointnet_Pointnet2_pytorch with only minor changes to work on the Paris Lille dataset.