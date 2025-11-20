# ------------------------------------------------------------
# evaluation_summary.py
# ------------------------------------------------------------
# Evaluates multiple trained models (pretrained & from-scratch)
# and saves a summary table in Excel (.xlsx) format.
# ------------------------------------------------------------
from imports import *
from config import SAVE_DIR
from data_loader import get_dataloaders
from model import build_model, get_device
from evaluate import evaluate_full


def evaluate_model_entry(model_path, pretrained, resolution, test_loader, device):
    """
    Load a trained model and compute evaluation metrics.
    Returns a dict entry for the results summary table.
    """
    model = build_model(pretrained=pretrained).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Evaluate full metrics (accuracy, precision, recall, F1, inference time)
    results = evaluate_full(model, test_loader, device)
    results["Model"] = f"ResNet18 ({'pretr' if pretrained else 'from'})"
    results["Resolution"] = f"{resolution}x{resolution}"
    return results


def main():
    print("==================================================")
    print("📊 Evaluation Summary Generator (Excel only)")
    print("==================================================\n")

    device = get_device()

    # Use test data (no augmentation)
    _, test_loader = get_dataloaders(use_augmentation=False)

    # ------------------------------------------------------------
    # Define your trained models (add more if you trained different sizes)
    # ------------------------------------------------------------
    model_entries = [
        (f"{SAVE_DIR}/resnet18_pretrained_final_256.pth", True, 256),
        (f"{SAVE_DIR}/resnet18_scratch_final_256", False, 256),
        # Add other resolutions here if trained:
        # (f"{SAVE_DIR}/resnet18_pretrained_512_best.pt", True, 512),
    ]

    # ------------------------------------------------------------
    # Evaluate each model and collect results
    # ------------------------------------------------------------
    all_results = []
    for path, pre, res in model_entries:
        if not os.path.exists(path):
            print(f"⚠️ Skipping missing model: {path}")
            continue
        print(f"Evaluating: {path}")
        result_entry = evaluate_model_entry(path, pre, res, test_loader, device)
        all_results.append(result_entry)

    if not all_results:
        print("❌ No models found for evaluation.")
        return

    # ------------------------------------------------------------
    # Convert to DataFrame and save to Excel
    # ------------------------------------------------------------
    df = pd.DataFrame(all_results, columns=[
        "Model", "Resolution", "Accuracy", "Precision", "Recall", "F1-Score", "Inference Time"
    ])

    # Round values for readability
    df = df.round({
        "Accuracy": 2, "Precision": 2, "Recall": 2, "F1-Score": 2, "Inference Time": 2
    })

    print("\n✅ Evaluation Summary Table:")
    print(df)

    # Save only to Excel
    excel_path = os.path.join(SAVE_DIR, "evaluation_summary.xlsx")
    df.to_excel(excel_path, index=False)

    print(f"\n💾 Saved evaluation summary to Excel:\n - {excel_path}")


if __name__ == "__main__":
    main()
