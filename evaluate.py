# ------------------------------------------------------------
# evaluate.py
# ------------------------------------------------------------
# Contains model evaluation:
#  - Accuracy, Precision, Recall, F1
#  - Inference time (GPU-synchronized)
#  - Confusion matrix saved as PNG
#  - Summary saved to Excel
# ------------------------------------------------------------

from imports import *
from config import CLASSES                      # class names from config.py
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, test_loader, test_dataset,
                   model_name="ResNet18",
                   resolution=224,
                   save_dir="/work/yazelelew_phd/Tooth/ModelsScript"):
    """
    Evaluate a trained model and append results to evaluation_summary.xlsx.
    
    Includes:
      - Accuracy, Precision, Recall, F1
      - Inference time (GPU synchronized)
      - Confusion matrix saved as PNG
    """

    model.eval()
    all_preds, all_labels = [], []
    forward_time = 0.0

    # ---------------------- Evaluation loop ----------------------
    with torch.no_grad():
        for images, labels in test_loader:

            images, labels = images.to(device), labels.to(device)

            # --- Proper CUDA timing (must sync before & after) ---
            if device.type == "cuda":
                torch.cuda.synchronize()

            start_t = time.time()
            outputs = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()

            forward_time += (time.time() - start_t)

            # Predictions
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Average time per image
    avg_time_per_image = forward_time / len(test_dataset)

    # ---------------------- Metrics ----------------------
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro')
    rec  = recall_score(all_labels, all_preds, average='macro')
    f1   = f1_score(all_labels, all_preds, average='macro')

    # ---------------------- Print metrics ----------------------
    print(f"\n📊 Evaluation — {model_name} ({resolution}x{resolution})")
    print(f"Accuracy:      {acc*100:.2f}%")
    print(f"Precision:     {prec*100:.2f}%")
    print(f"Recall:        {rec*100:.2f}%")
    print(f"F1-Score:      {f1*100:.2f}%")
    print(f"Avg forward-pass time: {avg_time_per_image:.6f} sec/img")

    # ---------------------- Confusion Matrix ----------------------
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

    print(f"📊 Confusion matrix saved → {cm_path}")

    # ---------------------- Excel logging ----------------------
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
    results_path = os.path.join(save_dir, "evaluation_summary.xlsx")

    if os.path.exists(results_path):
        df_existing = pd.read_excel(results_path)
        df_new = pd.concat([df_existing, pd.DataFrame([record])], ignore_index=True)
        df_new.to_excel(results_path, index=False)
    else:
        pd.DataFrame([record]).to_excel(results_path, index=False)

    print(f"💾 Saved summary → {results_path}")

    return record
