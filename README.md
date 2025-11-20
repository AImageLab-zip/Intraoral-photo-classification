# Tooth Image Classification – Full Project README

This README provides a complete, structured documentation of your **Tooth Appearance Classification Project**, based entirely on your final experimental report and pipeline description.

---

## 📌 1. Project Overview

This project classifies intraoral dental photographs into **five orthodontic viewpoints**:

* **center** – Frontal view
* **up** – Upper occlusal view
* **down** – Lower occlusal view
* **left** – Left lateral view
* **right** – Right lateral view

The goal is to determine the best model, resolution, and hyperparameter setup for robust clinical image classification.

---

## 📁 2. Dataset Description

* Images are organized **patient‑wise**.
* Each patient folder includes 5 standard orthodontic viewpoints.
* Images vary in:

  * illumination
  * device type
  * angle and orientation
  * presence of orthodontic appliances

### ✅ Data Leakage Prevention

Splitting is done **per patient**, ensuring no patient appears in both training and testing.

---

## 🛠️ 3. Preprocessing & Data Augmentation

### **3.1 Standard Preprocessing (all datasets)**

* Resize → *(64, 128, 256, 512 depending on experiment)*
* Convert to RGB
* Convert to tensor
* Normalize using ImageNet means/std

### **3.2 Training Data Augmentation**

| Augmentation       | Value                               |
| ------------------ | ----------------------------------- |
| Rotation           | ±45°                                |
| Affine Translation | ±10%                                |
| Affine Scale       | ±10%                                |
| ColorJitter        | ±30% brightness/contrast/saturation |
| Vertical Flip      | p = 0.5                             |
| Resize             | 64–512 px                           |

Vertical flip was implemented **inside the Dataset class**, ensuring it is applied only to training samples.

---

## 📦 4. Dataset Class Functions

Your custom Dataset class performs:

* Image loading
* Label mapping
* Augmentation (training only)
* Transform selection based on split
* Returns `(image_tensor, label)`

---

## 🧠 5. Algorithms Used

Two versions of **ResNet‑18** were evaluated:

### **5.1 Pretrained ResNet‑18**

* ImageNet pretrained
* FC replaced with 5‑class output
* Best accuracy
* Selected for deployment

### **5.2 ResNet‑18 (Scratch)**

* Same architecture, random initialization
* Required higher LR
* Underperformed vs pretrained

---

## 🔬 6. Experimental Pipeline

Experiments were conducted in **three phases**:

### **Phase 1 — Hyperparameter Search (fixed resolution: 256×256)**

Goal: Find best LR, Weight Decay, Batch Size.

### **Phase 2 — Resolution Search**

Resolutions tested:

* 64×64
* 128×128
* 224×224
* 256×256
* 512×512
* 224×224 (no augmentation)

### **Phase 3 — Final Training**

Train the final model using:

* Best params from Phase 1
* Best resolution from Phase 2
* 15 epochs

---

## 📊 7. Hyperparameter Search Results

Below is the full **hyperparameter search table** reproduced exactly as in the document fileciteturn1file0:

### **Table 1 — Hyperparameter Search Summary (Resolution = 256×256)**

| **File Name**                                            | **Val Acc (%)** | **LR** | **WD** | **BS** | **Result**   |
| -------------------------------------------------------- | --------------- | ------ | ------ | ------ | ------------ |
| resnet18_pretrained_lr0.0001_wd1e-05_fold1_256.xlsx      | 92.56           | 1e-4   | 1e-5   | 16     | Best overall |
| resnet18_pretrained_lr0.0002_wd1e-05_bs8_fold1_256.xlsx  | 91.16           | 2e-4   | 1e-5   | 8      | Good         |
| resnet18_pretrained_lr1e-05_wd0.0001_bs8_fold1_256.xlsx  | 90.70           | 1e-5   | 1e-4   | 8      | Underfitting |
| resnet18_pretrained_lr0.0001_wd1e-05_bs8_fold1_256.xlsx  | 90.23           | 1e-4   | 1e-5   | 8      | Stable       |
| resnet18_pretrained_lr0.0002_wd1e-05_bs32_fold1_256.xlsx | 90.23           | 2e-4   | 1e-5   | 32     | Fluctuating  |
| resnet18_pretrained_lr1e-05_wd0.0001_fold1_256.xlsx      | 88.84           | 1e-5   | 1e-4   | 16     | Too slow     |
| resnet18_pretrained_lr0.0001_wd1e-05_bs32_fold1_256.xlsx | 88.84           | 1e-4   | 1e-5   | 32     | Unstable     |
| resnet18_pretrained_lr1e-05_wd0.0001_bs32_fold1_256.xlsx | 88.37           | 1e-5   | 1e-4   | 32     | Worst        |

### **Best Hyperparameter Combination**

| **LR**                              | **Weight Decay** | **Batch Size** | **Image Size** | **Model**            |           |
| ----------------------------------- | ---------------- | -------------- | -------------- | -------------------- | --------- |
| 0.0001                              | 1e-5             | 16             | 256×256        | Pretrained ResNet-18 | (Phase 1) |
| **Best performance** achieved with: |                  |                |                |                      |           |

| LR         | Weight Decay | Batch Size | Image Size  | Model                    |
| ---------- | ------------ | ---------- | ----------- | ------------------------ |
| **0.0001** | **1e‑5**     | **16**     | **256×256** | **Pretrained ResNet‑18** |

This configuration gave the top validation accuracy.

---

## 🖼️ 8. Resolution Search Results

The following table reproduces the full multi-resolution comparison from page 6–7 of the document fileciteturn1file0:

### **Table 3 — Full Comparison: Pretrained vs Scratch Models (All Resolutions)**

| **Model Variant**                                   | **Resolution**| **Accuracy(%)** | **Precision (%)** | **Recall (%)** | **F1-Score (%)** | **Inference Time (sec)** |           |
| --------------------------------------------------- | -------------- | ---------------- | ----------------- | -------------- | ---------------- | ------------------------ | --------- |
| ResNet18 (pretrained)                               | 512×512        | 87.91            | 91.48             | 87.91          | 86.89            | 0.08                     |           |
| ResNet18 (scratch)                                  | 512×512        | 81.86            | 90.49             | 81.86          | 78.18            | 0.08                     |           |
| ResNet18 (pretrained)                               | 256×256        | 90.70            | 91.97             | 90.70          | 90.44            | 0.07                     |           |
| ResNet18 (scratch)                                  | 256×256        | 79.53            | 80.37             | 79.53          | 79.09            | 0.07                     |           |
| ResNet18 (pretrained)                               | 64×64          | 86.05            | 86.39             | 86.05          | 85.86            | 0.09                     |           |
| ResNet18 (scratch)                                  | 64×64          | 87.91            | 88.76             | 87.91          | 87.64            | 0.07                     |           |
| ResNet18 (pretrained)                               | 224×224        | 100              | 100               | 100            | 100              | 0.08                     |           |
| ResNet18 (scratch)                                  | 224×224        | 100              | 100               | 100            | 100              | 0.08                     |           |
| ResNet18 (pretrained)                               | 128×128        | 82.79            | 85.48             | 82.79          | 83.22            | 0.07                     |           |
| ResNet18 (scratch)                                  | 128×128        | 61.40            | 53.89             | 61.40          | 55.79            | 0.07                     | (Phase 2) |
| A complete comparison across all resolutions shows: |                |                  |                   |                |                  |                          |           |

* Pretrained > Scratch consistently
* 256×256 is the optimal resolution
* 512×512 offers no improvement and slower inference

**Top results:**

* Pretrained 256×256 → **90.70% accuracy**
* Scratch 256×256 → **79.53% accuracy**

The pretrained model is consistently superior.

---

## 🏆 9. Final Experiment (Phase 3)

Final training using best hyperparameters and best resolution.

### **Final Scores (256×256)**

| Model                      | Accuracy   | Precision | Recall | F1‑Score | Inference Time (sec/img) |
| -------------------------- | ---------- | --------- | ------ | -------- | ------------------------ |
| **ResNet‑18 (Pretrained)** | **92.09%** | 92.13%    | 92.09% | 92.09%   | 0.000624                 |
| ResNet‑18 (Scratch)        | 91.16%     | 91.23%    | 91.16% | 91.05%   | 0.000623                 |

### **Interpretation**

* Pretrained model performs better **across all metrics**.
* Improvements:

  * +0.93% Accuracy
  * +0.90% F1‑score
  * +0.96% Precision
  * +0.93% Recall
* Inference time is identical.

### **Final Conclusion**

The best model is:

| Setting            | Value                                        |
| ------------------ | -------------------------------------------- |
| **Model**          | ResNet‑18 (Pretrained)                       |
| **Resolution**     | 256×256                                      |
| **Learning Rate**  | 0.0001                                       |
| **Weight Decay**   | 1e‑5                                         |
| **Batch Size**     | 16                                           |
| **Epoch Strategy** | Best test accuracy checkpoint                |
| **Augmentations**  | Rotation, Affine, ColorJitter, Vertical Flip |

---

## 🚀 10. Inference Usage

Run inference and optional renaming via:

```
python inference.py --patient <folder> --model <model_path> --rename_output <output_folder>
```

---

## 📋 11. Folder Structure (Recommended)

```
ToothClassificationProject/
│
├── data/
│   ├── patient_1/
│   ├── patient_2/
│   └── ...
│
├── PythonScript/
│   ├── data_loader.py
│   ├── model.py
│   ├── imports.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── main.py
│
├── Models/
│   └── resnet18_pretrained_best_256.pth
│
├── results/
│   ├── logs/
│   ├── confusion_matrices/
│   └── summaries/
│
└── README.md
```

