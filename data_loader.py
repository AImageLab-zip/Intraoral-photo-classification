from torch.utils.data import Dataset, DataLoader
import os
import random
from glob import glob
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image

from config import (
    CLASSES, TEST_SIZE, RANDOM_STATE,
    BATCH_SIZE, NUM_WORKERS,
    IMAGENET_MEAN, IMAGENET_STD
)
def collect_data(root_dir):
 
    patients, image_paths, labels = [], [], []
    for patient_path in glob(os.path.join(root_dir, "*")):
        if not os.path.isdir(patient_path):
            continue
        patient_id = os.path.basename(patient_path)
        photos_dir = os.path.join(patient_path, "intraoral-photos")
        if not os.path.exists(photos_dir):
            print(f" Skipping {patient_id}: no intraoral-photos folder.")
            continue
        for file_name in os.listdir(photos_dir):
            lower = file_name.lower()

            if lower.endswith((".jpg", ".jpeg", ".png")):
                for cls in CLASSES:
                    if cls in lower:
                        image_paths.append(os.path.join(photos_dir, file_name))
                        labels.append(cls)
                        patients.append(patient_id)
                        break

    print(f" Found {len(image_paths)} labeled images from {len(set(patients))} patients.")
    return patients, image_paths, labels
def split_patients(patients):
    train_patients, test_patients = train_test_split(
        list(set(patients)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True
    )
    return train_patients, test_patients

def build_samples(patients, image_paths, labels, train_patients, test_patients):
    train_samples, test_samples = [], []
    for p, img, lbl in zip(patients, image_paths, labels):
        if p in train_patients:
            train_samples.append((img, lbl))
        else:
            test_samples.append((img, lbl))
    return train_samples, test_samples



def get_train_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(45),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
        transforms.RandomApply([transforms.RandomAdjustSharpness(2)], p=0.3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

def get_test_transform(img_size):

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
class ToothDataset(Dataset):

    def __init__(self, samples, transform=None, training=False):
        self.samples = samples
        self.transform = transform
        self.training = training
        self.class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.training and label in ["up", "down"]:
            if random.random() < 0.5:
                image = transforms.functional.vflip(image)
        if self.transform:
            image = self.transform(image)
        return image, self.class_to_idx[label]


def get_dataloaders(root_dir, img_size, use_augmentation=True):
    patients, image_paths, labels = collect_data(root_dir)
    train_patients, test_patients = split_patients(patients)
    train_samples, test_samples = build_samples(
        patients, image_paths, labels,
        train_patients, test_patients
    )
    if use_augmentation:
        train_transform = get_train_transform(img_size)
        test_transform = get_test_transform(img_size)
        mode = "Augmented"
    else:
        test_transform = get_test_transform(img_size)
        train_transform = test_transform
        mode = "Pure deterministic"
    train_dataset = ToothDataset(train_samples, transform=train_transform, training=True)
    test_dataset = ToothDataset(test_samples, transform=test_transform, training=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    print(f" {mode} dataloaders created — {len(train_samples)} train, {len(test_samples)} test samples.")
    return train_loader, test_loader, train_dataset, test_dataset
