import os
import torch
import wandb

from config import (
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    WANDB_PROJECT, DEVICE
)

from data_loader import get_dataloaders
from model import build_model, setup_training
from train import train_model
from evaluate import evaluate_model


def main():
    ROOT_DIR = "/work/yazelelew_phd/Tooth/FerraraDump"
    SAVE_DIR = "/work/yazelelew_phd/Tooth/ModelsScript"
    IMG_SIZE = 256
    USE_PRETRAINED = False

    MODEL_NAME = "resnet18_pretrained" if USE_PRETRAINED else "resnet18_scratch"

    wandb.init(
        project=WANDB_PROJECT,
        name=f"{MODEL_NAME}_Training_Run",
        config={
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "img_size": IMG_SIZE,
            "pretrained": USE_PRETRAINED
        }
    )
    train_loader, test_loader, train_dataset, test_dataset = get_dataloaders(
        root_dir=ROOT_DIR,
        img_size=IMG_SIZE,
        use_augmentation=True
    )
    print(f"\nTraining model | pretrained={USE_PRETRAINED}")

    model = build_model(pretrained=USE_PRETRAINED).to(DEVICE)
    criterion, optimizer = setup_training(model)
    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        model_name=MODEL_NAME,
        img_size=IMG_SIZE,
        save_dir=SAVE_DIR
    )

    best_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_best_{IMG_SIZE}.pth")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    evaluate_model(
        model=model,
        test_loader=test_loader,
        test_dataset=test_dataset,
        model_name=MODEL_NAME,
        resolution=IMG_SIZE,
        save_dir=SAVE_DIR
    )

    wandb.finish()
    print(f"\nDone! Results saved in {SAVE_DIR}")


if __name__ == "__main__":
    main()
