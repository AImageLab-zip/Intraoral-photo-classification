# ------------------------------------------------------------
# config.py (FINAL CLEAN VERSION)
# ------------------------------------------------------------

import torch

# ==== Classes ====
CLASSES = ["center", "down", "left", "right", "up"]

# ==== DataLoader settings ====
BATCH_SIZE = 8
NUM_WORKERS = 4
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ==== Training hyperparameters ====
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# ==== Image normalization ====
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ==== Device ====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==== WandB ====
WANDB_PROJECT = "Tooth_Classification1"

print(f"Config loaded | Device={DEVICE}")
