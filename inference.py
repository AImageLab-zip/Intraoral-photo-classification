import os
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import shutil

CLASSES = ["center", "down", "left", "right", "up"]
NUM_CLASSES = len(CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
def load_resnet18(model_path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(model, img_path):

    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, idx = torch.max(probs, 0)

    return CLASSES[idx], float(conf * 100)

def classify_folder(model, folder):
    results = {}
    for fname in os.listdir(folder):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(folder, fname)
        pred, conf = predict_image(model, img_path)

        results[fname] = {
            "filename": fname,
            "path": img_path,
            "prediction_raw": pred,
            "confidence": conf,
            "prediction_corrected": pred,
            "corrected": False
        }
    return results

def fix_class_conflicts(results):
    r = dict(results)
    counts = {c: 0 for c in CLASSES}

    for item in r.values():
        counts[item["prediction_corrected"]] += 1

    missing = [c for c in CLASSES if counts[c] == 0]
    duplicated = [c for c in CLASSES if counts[c] > 1]

    if not missing or not duplicated:
        return r

    for miss in missing:
        dup = duplicated[0]
        candidates = sorted(
            [x for x in r.values() if x["prediction_corrected"] == dup],
            key=lambda x: x["confidence"]
        )
        if len(candidates) > 1:
            worst = candidates[0]
            fname = worst["filename"]
            r[fname]["prediction_corrected"] = miss
            r[fname]["corrected"] = True

    return r
def sort_images(results, sort_dir):
    for cls in CLASSES:
        os.makedirs(os.path.join(sort_dir, cls), exist_ok=True)

    for item in results.values():
        dst = os.path.join(sort_dir, item["prediction_corrected"], item["filename"])
        shutil.copy(item["path"], dst)

def rename_images(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for item in results.values():
        predicted = item["prediction_corrected"]
        new_name = f"{predicted}.png"
        dst = os.path.join(output_dir, new_name)
        shutil.copy(item["path"], dst)

def main():
    parser = argparse.ArgumentParser(description="Intraoral-photo-classification Inference")
    parser.add_argument("--patient", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sort_output", default=None)
    parser.add_argument("--rename_output", default=None)
    args = parser.parse_args()

    model = load_resnet18(args.model)
    print("Model loaded on:", device)

    results = classify_folder(model, args.patient)
    corrected = fix_class_conflicts(results)

    df = pd.DataFrame(corrected).T

    os.makedirs(args.output, exist_ok=True)
    df.to_csv(os.path.join(args.output, "corrected_predictions.csv"))
   
    print("Saved  CSV .")

    if args.sort_output:
        os.makedirs(args.sort_output, exist_ok=True)
        sort_images(corrected, args.sort_output)
        print("Images sorted into:", args.sort_output)

    if args.rename_output:
        os.makedirs(args.rename_output, exist_ok=True)
        rename_images(corrected, args.rename_output)
        print("Images renamed into:", args.rename_output)

    print("Done.")

if __name__ == "__main__":
    main()
