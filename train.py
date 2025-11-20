import torch
import os
import time
import pandas as pd
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


def train_model(model, train_loader, test_loader, criterion, optimizer,
                num_epochs=15, model_name="resnet18", img_size=256,
                save_dir="/work/yazelelew_phd/Tooth/ModelsScript"):

    os.makedirs(save_dir, exist_ok=True)
    since = time.time()
    best_acc = 0.0
    history = []

    for epoch in range(num_epochs):

        # ---------------------------
        # TRAIN PHASE
        # ---------------------------
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_train += labels.size(0)

        epoch_loss = running_loss / total_train
        epoch_acc = (running_corrects.double() / total_train).item()

        # ---------------------------
        # EVALUATION PHASE
        # ---------------------------
        model.eval()
        running_corrects_test = 0
        total_test = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                running_corrects_test += torch.sum(preds == labels.data)
                total_test += labels.size(0)

        test_acc = (running_corrects_test.double() / total_test).item()

        # Save epoch history
        history.append({
            "Epoch": epoch + 1,
            "Train_Loss": epoch_loss,
            "Train_Acc": epoch_acc * 100,
            "Test_Acc": test_acc * 100
        })

        # ---------------------------
        # LOG METRICS TO WANDB  ⭐⭐
        # ---------------------------
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_accuracy": epoch_acc * 100,
            "test_accuracy": test_acc * 100,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "best_accuracy": best_acc * 100
        })

        # ---------------------------
        # SAVE BEST MODEL
        # ---------------------------
        if test_acc > best_acc:
            best_acc = test_acc
            best_path = os.path.join(save_dir, f"{model_name}_best_{img_size}.pth")
            torch.save(model.state_dict(), best_path)
            print(f"💾 Saved BEST model at epoch {epoch+1} (Test Acc = {test_acc*100:.2f}%) → {best_path}")

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_loss:.4f}  "
              f"Train Acc: {epoch_acc*100:.2f}%  "
              f"Test Acc: {test_acc*100:.2f}%")

    # ---------------------------
    # AFTER TRAINING
    # ---------------------------
    elapsed = time.time() - since
    print(f"\n🏁 Training finished in {elapsed/60:.1f} min.")
    print(f"🌟 BEST Epoch Accuracy: {best_acc*100:.2f}%")

    # Save training log (Excel)
    df_history = pd.DataFrame(history)
    excel_path = os.path.join(save_dir, f"{model_name}_trainlog_{img_size}.xlsx")
    df_history.to_excel(excel_path, index=False)
    print(f"📊 Training log saved: {excel_path}")

    # Save final model
    final_path = os.path.join(save_dir, f"{model_name}_final_{img_size}.pth")
    torch.save(model.state_dict(), final_path)
    print(f"💾 Final checkpoint saved: {final_path}")

    # Full checkpoint
    full_ckpt_path = os.path.join(save_dir, f"{model_name}_final_full_{img_size}.pth")
    torch.save({
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_function": str(criterion),
        "final_test_acc": test_acc,
        "img_size": img_size,
        "train_time_min": elapsed / 60
    }, full_ckpt_path)
    print(f"💾 Full checkpoint saved: {full_ckpt_path}")

    return model, df_history
