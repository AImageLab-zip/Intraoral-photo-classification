
from imports import *
from config import CLASSES, LEARNING_RATE, WEIGHT_DECAY


# ------------------------------------------------------------
# STEP 1: Device selection
# ------------------------------------------------------------
def get_device():
    """
    Return CUDA if available, otherwise CPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    return device


# ------------------------------------------------------------
# STEP 2: Model builder
# ------------------------------------------------------------
def build_model(pretrained=True, num_classes=len(CLASSES)):
    """
    Build a ResNet18 model, optionally using pretrained ImageNet weights.

    Args:
        pretrained (bool): Use pretrained weights if True.
        num_classes (int): Number of output classes.

    Returns:
        model (torch.nn.Module): The initialized ResNet18 model.
    """
    model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    print(f"🧠 Built ResNet18 | Pretrained={pretrained} | Classes={num_classes}")
    return model


# ------------------------------------------------------------
# STEP 3: Training setup (optimizer & loss)
# ------------------------------------------------------------
def setup_training(model, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
    """
    Create loss function and optimizer for the model.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return criterion, optimizer
