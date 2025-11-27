# Intraoral-photo-classification

This repository provides a deep learning system for classifying intraoral dental photographs into five standard orthodontic viewpoints:

- center
- up (upper occlusal)
- down (lower occlusal)
- left
- right

The project evaluates preprocessing, data augmentation strategies, multiple image resolutions, and two versions of ResNet-18 (pretrained and trained from scratch).

## 1. Installation

Install dependencies:

```
pip install -r requirements.txt
```

The framework automatically uses GPU if available; otherwise CPU is used.


## 2. for Retraining

Run training:

```
python main.py
```

## 3. Inference for use the trained model

```
python inference.py     --patient <path_to_patient_folder>     --model <best_model.pth>     --output <results_folder>
```

## 4. Model Architecture

The classifier is based on ResNet-18 both pretrained and from Scratch.  
The final fully connected layer is replaced with a 5-class output.


## 5. Experimental Summary

Final evaluation:

| Model               | Accuracy | Precision | Recall | F1‑Score | Inference Time (sec/img) |
|---------------------|----------|-----------|--------|----------|----------------------------|
| ResNet18 Pretrained** |92.09% |92.15%** | 92.09% | 92.08%** | 0.000732 |
| ResNet18 Scratch    | 86.98%  | 88.06%    | 86.98% | 86.46%   | 0.000760 |


## 6. Model Link
Download the model Here: https://drive.google.com/drive/u/0/folders/1LM9fIciXma-2ak9nyRqOzRrinzIH7K0S
## 7. Requirements

torch
torchvision
pillow
pandas
numpy
shutil
argparse
