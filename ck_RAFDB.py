# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from timm import create_model

# Configurations
MODEL_PATH = "/path/to/DenseNet121_f1.pth"  # Cập nhật đường dẫn tới checkpoint ResNet50 đã huấn luyện
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 7
TARGET_SIZE = (224, 224)  # Thay đổi từ (300, 300) để phù hợp với ResNet50 trong mã thứ hai
BATCH_SIZE = 32

# Label mapping from CK+ to RAF-DB (excluding Contempt)
ck_to_raf_mapping = {
    0: 5,  # anger -> angry
    1: 2,  # disgust -> disgust
    2: 1,  # fear -> fear
    3: 3,  # happiness -> happy
    4: 4,  # sadness -> sad
    5: 0,  # surprise -> surprise
    6: 6,  # neutral -> neutral
    # 7: excluded (contempt)
}

label_to_emotion = {
    0: "surprise",
    1: "fear",
    2: "disgust",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "neutral",
}

# Transformations (điều chỉnh để phù hợp với mã thứ hai)
test_transforms = transforms.Compose(
    [
        transforms.Resize(TARGET_SIZE),
        transforms.Grayscale(num_output_channels=3),  # Thêm để đồng bộ với mã thứ hai
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Custom Dataset for CK+ Extended
class CKPlusDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        # Filter out Contempt (label 7)
        self.data = self.data[self.data["emotion"] != 7]
        # Map CK+ labels to RAF-DB labels
        self.data["emotion"] = self.data["emotion"].map(ck_to_raf_mapping)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pixels = self.data.iloc[idx]["pixels"]
        emotion = self.data.iloc[idx]["emotion"]

        # Convert pixel string to numpy array and reshape to 48x48
        pixel_values = np.array([int(p) for p in pixels.split()], dtype=np.uint8)
        image = pixel_values.reshape(48, 48)

        # Convert to PIL image (grayscale)
        from PIL import Image
        image = Image.fromarray(image, mode="L").convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, emotion

# Load the dataset
ck_dataset = CKPlusDataset(
    csv_file="/kaggle/input/ckdataset/ckextended.csv", transform=test_transforms
)
ck_loader = DataLoader(
    ck_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)

# Define the EfficientNet-B4 model (lấy từ mã thứ hai)
# Define the ResNet50 model
class DenseNet121_Model(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet121_Model, self).__init__()
        self.backbone = timm.create_model(
            "densenet121", pretrained=True, num_classes=num_classes
        )

    def forward(self, x):
        return self.backbone(x)
# Load the model
model = DenseNet121_Model(num_classes=NUM_CLASSES)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
state_dict = {
    k.replace("module.", ""): v for k, v in state_dict.items()
}  # Remove "module." prefix if trained with DataParallel
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

# Prediction
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in ck_loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute Confusion Matrix (raw counts)
cm_raw = confusion_matrix(all_labels, all_preds)

# Compute Normalized Confusion Matrix (row-wise normalization)
cm_normalized = cm_raw.astype("float") / cm_raw.sum(axis=1)[:, np.newaxis]
cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0 if any row sums to 0

# Visualize Raw Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_raw,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[label_to_emotion[i] for i in range(NUM_CLASSES)],
    yticklabels=[label_to_emotion[i] for i in range(NUM_CLASSES)],
)
plt.title("Raw Confusion Matrix on CK+ Extended Dataset", fontsize=14)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("True", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(
    "DenseNet121_confusion_matrix_ckextended_raw.png",
    dpi=300,
    bbox_inches="tight",
)
print(
    "Raw confusion matrix saved to 'DenseNet121_confusion_matrix_ckextended_raw.png'"
)

# Visualize Normalized Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=[label_to_emotion[i] for i in range(NUM_CLASSES)],
    yticklabels=[label_to_emotion[i] for i in range(NUM_CLASSES)],
)
plt.title("Normalized Confusion Matrix on CK+ Extended Dataset", fontsize=14)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("True", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(
    "DenseNet121_confusion_matrix_ckextended_normalized.png",
    dpi=300,
    bbox_inches="tight",
)
print(
    "Normalized confusion matrix saved to 'DenseNet121_confusion_matrix_ckextended_normalized.png'"
)

# Save both matrices to CSV
cm_raw_df = pd.DataFrame(
    cm_raw,
    index=[label_to_emotion[i] for i in range(NUM_CLASSES)],
    columns=[label_to_emotion[i] for i in range(NUM_CLASSES)],
)
cm_raw_df.to_csv("DenseNet121_confusion_matrix_ckextended_raw.csv")
print(
    "Raw confusion matrix saved to 'DenseNet121_confusion_matrix_ckextended_raw.csv'"
)

cm_normalized_df = pd.DataFrame(
    cm_normalized,
    index=[label_to_emotion[i] for i in range(NUM_CLASSES)],
    columns=[label_to_emotion[i] for i in range(NUM_CLASSES)],
)
cm_normalized_df.to_csv(
    "DenseNet121_confusion_matrix_ckextended_normalized.csv"
)
print(
    "Normalized confusion matrix saved to 'DenseNet121_confusion_matrix_ckextended_normalized.csv'"
)

# Print some basic statistics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="macro")
prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)

print(f"Accuracy: {acc:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")