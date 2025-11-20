# ------------------------------------------------------------
# config.py
# ------------------------------------------------------------
# Stores project-wide constants and default parameters.
# ------------------------------------------------------------
from imports import *
# ==== Dataset paths ====
ROOT_DIR = "/work/yazelelew_phd/Tooth/FerraraDump"
SAVE_DIR = "/work/yazelelew_phd/Tooth/ModelsScript"

# ==== Classes ====
CLASSES = ["center", "down", "left", "right", "up"]

# ==== DataLoader settings ====
BATCH_SIZE = 8
NUM_WORKERS = 4
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ==== Image sizes ====
# You can change this in main.py to test multiple resolutions
IMG_SIZE = (256, 256)

# ==== Training hyperparameters ====
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 5   # early stopping patience (epochs)

# ==== Device ====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==== W&B project ====
WANDB_PROJECT = "Tooth_Classification1"

print(f"✅ Config loaded | Device: {DEVICE} | Image Size: {IMG_SIZE}")
