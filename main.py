
from imports import *
from config import SAVE_DIR, NUM_EPOCHS
from data_loader import get_dataloaders
from model import build_model, get_device, setup_training
from train import train_model
from evaluate import evaluate_model


def main():
    print("==================================================")
    print("🦷 Tooth Appearance Classification - Training Start")
    print("==================================================\n")

    # --------------------------------------------------
    # Initialize WandB (one run for entire experiment)
    # --------------------------------------------------
    wandb.init(
        project="Tooth_Classification_Final",
        name="ResNet18_Training_Run",
        config={
            "batch_size": 16,
            "epochs": NUM_EPOCHS,
            "learning_rate": 1e-4,
            "img_size": 256
        }
    )

    # --------------------------------------------------
    # STEP 1: Load Data
    # --------------------------------------------------
    train_loader, test_loader, train_dataset, test_dataset = get_dataloaders(
        use_augmentation=True
    )

    # --------------------------------------------------
    # STEP 2: Setup Device
    # --------------------------------------------------
    device = get_device()
    print(f"Using device: {device}")

    # ------------------------------------------------------------
    # TRAINING 1️⃣: Pretrained ResNet18
    # ------------------------------------------------------------
    print("\n🔥 Training ResNet18 (Pretrained)...")
    model_pretrained = build_model(pretrained=True).to(device)
    criterion_pre, optimizer_pre = setup_training(model_pretrained)

    model_pretrained, hist_pre = train_model(
        model=model_pretrained,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion_pre,
        optimizer=optimizer_pre,
        num_epochs=NUM_EPOCHS,
        model_name="resnet18_pretrained",
        img_size=256,
        save_dir=SAVE_DIR
    )

    # ------------------------------------------------------------
    # TRAINING 2️⃣: ResNet18 from Scratch
    # ------------------------------------------------------------
    print("\n🔥 Training ResNet18 (From Scratch)...")
    model_scratch = build_model(pretrained=False).to(device)
    criterion_scratch, optimizer_scratch = setup_training(model_scratch)

    model_scratch, hist_scratch = train_model(
        model=model_scratch,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion_scratch,
        optimizer=optimizer_scratch,
        num_epochs=NUM_EPOCHS,
        model_name="resnet18_scratch",
        img_size=256,
        save_dir=SAVE_DIR
    )

    # --------------------------------------------------
    # FINAL EVALUATION
    # --------------------------------------------------
    print("\n📊 Final Evaluation...")

    # Load best models
    pretrained_path = os.path.join(SAVE_DIR, "resnet18_pretrained_best_256.pth")
    scratch_path = os.path.join(SAVE_DIR, "resnet18_scratch_best_256.pth")

    # Evaluate Scratch Model
    print("\nEvaluating Scratch Model...")
    model_scratch_eval = build_model(pretrained=False).to(device)
    model_scratch_eval.load_state_dict(torch.load(scratch_path, map_location=device))

    evaluate_model(
        model_scratch_eval,
        test_loader,
        test_dataset,
        model_name="ResNet18_Scratch",
        resolution=256,
        save_dir=SAVE_DIR
    )

    # Evaluate Pretrained Model
    print("\nEvaluating Pretrained Model...")
    model_pre_eval = build_model(pretrained=True).to(device)
    model_pre_eval.load_state_dict(torch.load(pretrained_path, map_location=device))

    evaluate_model(
        model_pre_eval,
        test_loader,
        test_dataset,
        model_name="ResNet18_Pretrained",
        resolution=256,
        save_dir=SAVE_DIR
    )

    # --------------------------------------------------
    # END
    # --------------------------------------------------
    wandb.finish()

    print("\n==================================================")
    print("✅ Training and Evaluation completed successfully!")
    print(f"📁 Results saved in: {SAVE_DIR}")
    print("==================================================")


if __name__ == "__main__":
    main()
