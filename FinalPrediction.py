

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# -------------------------------------------------------
# 1️⃣ SETUP: define classes
# -------------------------------------------------------
CLASSES = ["up", "down", "left", "right", "center"]
NUM_CLASSES = len(CLASSES)

# -------------------------------------------------------
# 2️⃣ LOAD TRAINED MODEL (ResNet18)
# -------------------------------------------------------
def load_resnet18(model_path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

MODEL_PATH = "/work/yazelelew_phd/Tooth/ModelsScript/resnet18_pretrained_best_256.pth"
model = load_resnet18(MODEL_PATH)
print("✅ Model loaded successfully")

# -------------------------------------------------------
# 3️⃣ IMAGE TRANSFORM
# -------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------------------------------
# 4️⃣ SINGLE IMAGE PREDICTION
# -------------------------------------------------------
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, idx = torch.max(probs, 0)

    return CLASSES[idx], float(conf * 100)

# -------------------------------------------------------
# 5️⃣ PREDICT FOLDER
# -------------------------------------------------------
def classify_folder(folder_path):
    results = {}

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        path = os.path.join(folder_path, fname)
        pred, conf = predict_image(path)

        results[fname] = {
            "filename": fname,
            "path": path,
            "prediction_raw": pred,
            "confidence": conf,
            "prediction_corrected": pred,
            "corrected": False
        }

    return results

# -------------------------------------------------------
# 6️⃣ AUTO-CORRECTION LOGIC
# -------------------------------------------------------
def fix_class_conflicts(results_dict):
    results = dict(results_dict)

    # Count occurrences
    counts = {c: 0 for c in CLASSES}
    for item in results.values():
        counts[item["prediction_corrected"]] += 1

    # Find missing & duplicated classes
    missing = [c for c in CLASSES if counts[c] == 0]
    duplicated = [c for c in CLASSES if counts[c] > 1]

    # No corrections needed
    if not missing or not duplicated:
        return results

    # Fix duplicates → missing
    for missing_class in missing:
        dup_class = duplicated[0]

        # pick the weakest prediction to correct
        candidates = sorted(
            [item for item in results.values() if item["prediction_corrected"] == dup_class],
            key=lambda x: x["confidence"]
        )

        if len(candidates) > 1:
            weakest = candidates[0]
            fname = weakest["filename"]

            results[fname]["prediction_corrected"] = missing_class
            results[fname]["corrected"] = True

            counts[missing_class] += 1
            counts[dup_class] -= 1

    return results

# -------------------------------------------------------
# 7️⃣ RUN PIPELINE
# -------------------------------------------------------
FOLDER = "/work/yazelelew_phd/Tooth/ScriptPython/Dataset_CVPR2026_intraoral_photos/Dataset_CVPR2026_intraoral_photos/201_1979/intraoral_photos"

raw_results = classify_folder(FOLDER)
final_results = fix_class_conflicts(raw_results)

df = pd.DataFrame(final_results).T

# Print results
print("\n===== FINAL CORRECTED PREDICTIONS =====")
print(df)

# Show working directory
cwd = os.getcwd()
print("\nCurrent working directory:", cwd)

# Force-save Excel to a fixed directory (no ambiguity)
EXCEL_OUTPUT = "/work/yazelelew_phd/Tooth/ScriptPython/corrected_predictions.xlsx"

df.to_excel(EXCEL_OUTPUT)

print("\n✔ Excel saved at:", EXCEL_OUTPUT)
print("✔ Done.")
