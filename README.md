# The Role of Image Resolution in CNN Training

This repository contains the code and report for a university project investigating how image resolution affects the training efficiency and performance of convolutional neural networks (CNNs).

The project evaluates the trade-off between classification accuracy and computational cost when training identical CNN models on different image resolutions.

---

## Authors
- Bertram Sillesen (s255253)
- Lasse Lundbæk (s251099)
- Noah Westheimer (s255205)

---

## Project Description
Image resolution plays a crucial role in computer vision tasks. While higher resolutions preserve more visual detail, they also increase memory usage and training time.  
In this project, we train the same CNN architecture from scratch on a fruit image dataset using multiple image resolutions and compare:

- Training time
- Validation loss
- Test accuracy

The goal is to identify how resolution choice impacts model performance and practicality.

---

## Folder Descriptions

### `experiment/`
Contains training experiments conducted by each group member.  
Each subfolder includes scripts, logs, or checkpoints related to individual training runs.

### `final_model.py`
The finalized CNN model definition used for evaluation and comparison across different image resolutions.

### `architecture (onnx-converter)/`
Scripts and exported files related to converting the trained model to ONNX format.
- `export_onnx.py` exports the model
- `fruit_model_untrained_dynamic.onnx` contains the exported architecture

### `data analysis/`
Contains Jupyter notebooks and generated figures used for analysis and visualization.
- Training time vs. accuracy
- Resolution vs. accuracy
- Resolution vs. training time
- CI-intervals for each of the resolutions

### `graphs/`
Final plots used in the report.

---

## Requirements
The project was developed and tested using the following libraries:

- Python 3.x
- torch
- torchvision
- matplotlib

Optional (for analysis and plotting):
- jupyter
- numpy
