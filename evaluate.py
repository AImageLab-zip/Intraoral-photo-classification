
import torch
import pandas as pd
import os
import time
from config import CLASSES                      
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score, confusion_matrix
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, test_loader, test_dataset, model_name, resolution, save_dir):
    
    model.eval()
    all_preds, all_labels = [], []
    forward_time = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_t = time.time()
            outputs = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            forward_time += (time.time() - start_t)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_time_per_image = forward_time / len(test_dataset)
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro')
    rec  = recall_score(all_labels, all_preds, average='macro')
    f1   = f1_score(all_labels, all_preds, average='macro')
    print(f"\nEvaluation — {model_name} ({resolution}x{resolution})")
    print(f"Accuracy:      {acc*100:.2f}%")
    print(f"Precision:     {prec*100:.2f}%")
    print(f"Recall:        {rec*100:.2f}%")
    print(f"F1-Score:      {f1*100:.2f}%")
    print(f"Avg forward-pass time: {avg_time_per_image:.6f} sec/img")
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES,
                yticklabels=CLASSES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — {model_name}")

    cm_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f" Confusion matrix saved → {cm_path}")
    record = {
        "Model": model_name,
        "Resolution": f"{resolution}x{resolution}",
        "Accuracy": round(acc * 100, 2),
        "Precision": round(prec * 100, 2),
        "Recall": round(rec * 100, 2),
        "F1-Score": round(f1 * 100, 2),
        "Inference Time (sec)": round(avg_time_per_image, 6),
    }

    os.makedirs(save_dir, exist_ok=True)
    results_path = os.path.join(save_dir, "evaluation_summary.csv")

    if os.path.exists(results_path):
        df_existing = pd.read_csv(results_path)
        df_new = pd.concat([df_existing, pd.DataFrame([record])], ignore_index=True)
        df_new.to_csv(results_path, index=False)
    else:
        pd.DataFrame([record]).to_csv(results_path, index=False)

    print(f" Saved summary → {results_path}")

    return record
