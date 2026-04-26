# ARM-YOLOv12 Space Manipulator Vision and Control Package

This repository contains the code, dataset, trained model, control scripts, and experimental materials used for the journal submission. The package is organized with relative paths so it can be moved as a complete folder without changing hard-coded local paths.

For code debugging questions, please contact the author at mahandsome375@gmail.com.

## Package Contents

- `main.pdf`: manuscript file for the journal submission.
- `dataset/`: visual dataset used by the detector.
- `dataset.yaml`: dataset configuration file.
- `ultralytics/`: modified ARM-YOLOv12 detection code and related scripts.
- `best.pt`: trained detection weight used for inference and evaluation.
- `runs/`: saved training and comparative experimental results.
- `bridge/`: scripts for video processing, trajectory generation, coordinate export, and control-related data exchange.
- `udp_test/`: UDP communication test script.
- `control/`: controller scripts used with the visual detection results.
- `satellite_motion.mp4`: example video material.

## Dataset

The dataset is stored in YOLO format:

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
```

The dataset configuration is provided in `dataset.yaml`. A related dataset configuration is also included under `ultralytics/cfg/datasets/arm_dataset.yaml`.

## Vision Model

The vision part is based on a modified Ultralytics YOLOv12 codebase. The main project-specific files include:

- `ultralytics/train_arm.py`: training entry script for the ARM-YOLOv12 model.
- `ultralytics/depth_test.py`: visual inference and depth-related test script.
- `ultralytics/depth_test_second.py`: additional visual inference and depth-related test script.
- `ultralytics/predict_multi.py`: multi-image or multi-frame prediction script.
- `best.pt`: trained model checkpoint.

The saved training and evaluation outputs are included in `runs/`.

## Control Scripts

The control folder contains the controller scripts used in the project:

- `control/New_RL.m`: controller script associated with the visual-data-based control experiment.
- `control/PID.m`: ordinary PID controller baseline.

These MATLAB scripts are included as part of the control comparison materials for the journal package.

## Environment

The Python code uses the Ultralytics/PyTorch environment. The project metadata and Python dependency information are provided in `pyproject.toml`.

Recommended environment:

- Python 3.8 or later
- PyTorch
- OpenCV
- NumPy
- PyYAML
- Matplotlib
- SciPy
- MATLAB for the control scripts in `control/`

Install the Python package from the package root if needed:

```bash
pip install -e .
```

## Basic Usage

Run training from the package root:

```bash
python ultralytics/train_arm.py
```

Run visual inference or testing with the provided scripts:

```bash
python ultralytics/depth_test.py
python ultralytics/depth_test_second.py
python ultralytics/predict_multi.py
```

The provided trained model is `best.pt`. Keep the package directory structure unchanged so the relative paths in the scripts and configuration files continue to work.

For the control part, open MATLAB from the package root and run the scripts in `control/` as needed:

```text
control/PID.m
control/New_RL.m
```

## Notes for Journal Upload

This package includes the manuscript, dataset, modified detection code, trained model weight, experimental outputs, and control scripts. The directory structure should be kept unchanged when uploading or sharing the package.
