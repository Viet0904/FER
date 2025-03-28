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
import timm

# Configurations
MODEL_PATH = "/kaggle/working/MobileNetV2_RAFDB_vs1_f1.pth"  # Cập nhật đường dẫn nếu cần
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 7
TARGET_SIZE = (224, 224)
BATCH_SIZE = 32

# Label mapping from CK+ to RAF-DB
ck_to_raf_mapping = {0: 5, 1: 2, 2: 1, 3: 3, 4: 4, 5: 0, 6: 6}

label_to_emotion = {
    0: "surprise", 1: "fear", 2: "disgust", 3: "happy", 4: "sad", 5: "angry", 6: "neutral"
}

# Transformations
test_transforms = transforms.Compose(
    [
        transforms.Resize(TARGET_SIZE),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Custom Dataset for CK+ Extended
class CKPlusDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data["emotion"] != 7]
        self.data["emotion"] = self.data["emotion"].map(ck_to_raf_mapping)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pixels = self.data.iloc[idx]["pixels"]
        emotion = self.data.iloc[idx]["emotion"]
        pixel_values = np.array([int(p) for p in pixels.split()], dtype=np.uint8)
        image = pixel_values.reshape(48, 48)
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

# Define SEModule
class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Define MobileNetV2_Attention
class MobileNetV2_Attention(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2_Attention, self).__init__()
        self.backbone = timm.create_model(
            "mobilenetv2_100", pretrained=False, features_only=True  # Không dùng pretrained trong dự đoán
        )
        self.feature_info = self.backbone.feature_info

        # Thêm attention module sau lớp đặc trưng thứ 3
        self.attention = SEModule(self.feature_info.channels()[3])

        # Thêm Batch Normalization và Dropout
        self.bn = nn.BatchNorm2d(self.feature_info.channels()[-1])
        self.dropout = nn.Dropout(0.5)

        # Lớp pooling và phân loại
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_info.channels()[-1], num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        features[3] = self.attention(features[3])  # Áp dụng attention

        # Sử dụng lớp đặc trưng cuối cùng
        x = self.avgpool(features[-1])
        x = self.bn(x)  # Chuẩn hóa
        x = self.dropout(x)  # Dropout
        x = x.flatten(1)
        x = self.classifier(x)
        return x

# Load the model
model = MobileNetV2_Attention(num_classes=NUM_CLASSES)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

# Tính và in số lượng tham số của mô hình
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

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

# Compute Confusion Matrix
cm_raw = confusion_matrix(all_labels, all_preds)
cm_normalized = cm_raw.astype("float") / cm_raw.sum(axis=1)[:, np.newaxis]
cm_normalized = np.nan_to_num(cm_normalized)

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
    "MobileNetV2_RAFDB_vs1_confusion_matrix_ckextended_raw.png",
    dpi=300,
    bbox_inches="tight",
)
print(
    "Raw confusion matrix saved to 'MobileNetV2_RAFDB_vs1_confusion_matrix_ckextended_raw.png'"
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
    "MobileNetV2_RAFDB_vs1_confusion_matrix_ckextended_normalized.png",
    dpi=300,
    bbox_inches="tight",
)
print(
    "Normalized confusion matrix saved to 'MobileNetV2_RAFDB_vs1_confusion_matrix_ckextended_normalized.png'"
)

# Save both matrices to CSV
cm_raw_df = pd.DataFrame(
    cm_raw,
    index=[label_to_emotion[i] for i in range(NUM_CLASSES)],
    columns=[label_to_emotion[i] for i in range(NUM_CLASSES)],
)
cm_raw_df.to_csv("MobileNetV2_RAFDB_vs1_confusion_matrix_ckextended_raw.csv")
print(
    "Raw confusion matrix saved to 'MobileNetV2_RAFDB_vs1_confusion_matrix_ckextended_raw.csv'"
)

cm_normalized_df = pd.DataFrame(
    cm_normalized,
    index=[label_to_emotion[i] for i in range(NUM_CLASSES)],
    columns=[label_to_emotion[i] for i in range(NUM_CLASSES)],
)
cm_normalized_df.to_csv(
    "MobileNetV2_RAFDB_vs1_confusion_matrix_ckextended_normalized.csv"
)
print(
    "Normalized confusion matrix saved to 'MobileNetV2_RAFDB_vs1_confusion_matrix_ckextended_normalized.csv'"
)

# Print basic statistics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="macro")
prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)

print(f"Accuracy: {acc:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")