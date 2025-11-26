
from config import CLASSES, LEARNING_RATE, WEIGHT_DECAY, DEVICE
from torchvision import models
import torch.nn as nn
import torch.optim as optim
def build_model(pretrained=True, num_classes=len(CLASSES)):
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    print(f"Built ResNet18 | pretrained={pretrained} | classes={num_classes}")
    return model.to(DEVICE)
def setup_training(model, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    return criterion, optimizer
