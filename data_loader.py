# ------------------------------------------------------------
# data_loader.py
# ------------------------------------------------------------
# Handles dataset preparation, augmentation, and dataloaders
# for the Tooth Appearance Classification project.
# ------------------------------------------------------------

from imports import *
from config import ROOT_DIR, CLASSES, TEST_SIZE, RANDOM_STATE, BATCH_SIZE, NUM_WORKERS, IMG_SIZE


# ------------------------------------------------------------
# STEP 1: Collect image paths and labels
# ------------------------------------------------------------
def collect_data():
    """
    Collect image paths and labels from the structure:
        ROOT_DIR/patient_id/intraoral-photos/{center, down, left, right, up}.jpg
    Supports image formats: .jpg, .jpeg, .png (case-insensitive)
    """

    patients, image_paths, labels = [], [], []

    # Loop over each patient folder
    for patient_path in glob(os.path.join(ROOT_DIR, "*")):
        if not os.path.isdir(patient_path):
            continue

        patient_id = os.path.basename(patient_path)
        photos_dir = os.path.join(patient_path, "intraoral-photos")

        if not os.path.exists(photos_dir):
            print(f"⚠️ Skipping {patient_id}: no intraoral-photos folder.")
            continue

        # Loop over all files inside intraoral-photos
        for file_name in os.listdir(photos_dir):
            lower_name = file_name.lower()

            # ✅ Only process valid image formats
            if lower_name.endswith((".jpg", ".jpeg", ".png")):
                # Try to find which class this image belongs to
                for cls in CLASSES:
                    if cls in lower_name:
                        img_path = os.path.join(photos_dir, file_name)
                        image_paths.append(img_path)
                        labels.append(cls)
                        patients.append(patient_id)
                        break  # stop once class found

    print(f"✅ Found {len(image_paths)} labeled images from {len(set(patients))} patients.")
    return patients, image_paths, labels


# ------------------------------------------------------------
# STEP 2: Split into training and testing sets
# ------------------------------------------------------------
def split_patients(patients):
    """
    Split patients into training and testing groups (to avoid data leakage).
    """
    train_patients, test_patients = train_test_split(
        list(set(patients)), test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    return train_patients, test_patients


# ------------------------------------------------------------
# STEP 3: Build sample lists
# ------------------------------------------------------------
def build_samples(patients, image_paths, labels, train_patients, test_patients):
    """
    Map each patient to its corresponding images and split into train/test samples.
    """
    train_samples, test_samples = [], []

    for path, lbl, patient in zip(image_paths, labels, patients):
        if patient in train_patients:
            train_samples.append((path, lbl))
        else:
            test_samples.append((path, lbl))

    return train_samples, test_samples


# ------------------------------------------------------------
# STEP 4a: Training transform (with augmentation)
# ------------------------------------------------------------
def get_train_transform(img_size=256):
    """
    Data augmentation for training.
    Each image will be randomly transformed on-the-fly at every epoch.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),  # unify resolution
        transforms.RandomRotation(45),            # ±45° rotation
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),                # ±10% shift
            scale=(0.9, 1.1)                     # ±10% zoom
        ),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3         # ±30% brightness & contrast
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3)], p=0.3  # 30% chance of blur
        ),
        transforms.RandomApply(
            [transforms.RandomAdjustSharpness(2)], p=0.3      # 30% chance of sharpness tweak
        ),
        transforms.RandomHorizontalFlip(p=0.5),   # 50% chance to flip horizontally
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


# ------------------------------------------------------------
# STEP 4b: Test / Validation transform (no randomness)
# ------------------------------------------------------------
def get_test_transform(img_size=256):
    """
    Preprocessing for testing — deterministic, no random augmentations.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


# ------------------------------------------------------------
# STEP 4c: Pure transform (same for both train & test)
# ------------------------------------------------------------
def get_pure_transform(img_size=224):
    """
    Deterministic preprocessing used for both training and testing.
    This disables augmentation completely.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),                # Resize to uniform resolution
        transforms.ToTensor(),                                 # Convert to tensor [0,1]
        transforms.Normalize([0.485, 0.456, 0.406],            # Normalize to ImageNet stats
                             [0.229, 0.224, 0.225])
    ])


print("✅ Augmentation functions defined.")


# ------------------------------------------------------------
# STEP 5: Custom Dataset class
# ------------------------------------------------------------
class ToothDataset(Dataset):
    """
    Custom dataset for tooth appearance classification.
    Each sample: (image, label_index)
    """
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        self.class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        # ====================================================
        # ⭐ NEW: Apply vertical flip ONLY for up/down classes
        # ====================================================
        if label in ["up", "down"]:
            if random.random() < 0.5:     # 50% chance
                image = transforms.functional.vflip(image)
        # ====================================================

        if self.transform:
            image = self.transform(image)

        label_idx = self.class_to_idx[label]
        return image, label_idx



# ------------------------------------------------------------
# STEP 6: Build DataLoaders (toggle augmentation)
# ------------------------------------------------------------
def get_dataloaders(use_augmentation=True):
    """
    Return train_loader, test_loader, train_dataset, test_dataset.

    Args:
        use_augmentation (bool): if True → use rich augmentation for training.
                                 if False → use pure deterministic preprocessing for both.
    """

    # Step A — Collect all raw data
    patients, image_paths, labels = collect_data()

    # Step B — Split by patient (no leakage)
    train_patients, test_patients = split_patients(patients)

    # Step C — Build sample lists
    train_samples, test_samples = build_samples(
        patients, image_paths, labels,
        train_patients, test_patients
    )

    # Step D — Choose transforms
    if use_augmentation:
        train_transform = get_train_transform(img_size=IMG_SIZE[0])
        test_transform = get_test_transform(img_size=IMG_SIZE[0])
        mode = "Augmented"
    else:
        pure_transform = get_pure_transform(img_size=IMG_SIZE[0])
        train_transform = test_transform = pure_transform
        mode = "Pure deterministic"

    # Step E — Create Dataset objects
    train_dataset = ToothDataset(train_samples, transform=train_transform)
    test_dataset = ToothDataset(test_samples, transform=test_transform)

    # Step F — Create DataLoaders
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

    print(f"✅ {mode} dataloaders created — {len(train_samples)} train, {len(test_samples)} test samples.")

    # ⭐ NEW: return datasets as well!
    return train_loader, test_loader, train_dataset, test_dataset