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
MODEL_PATH = "ResNet50_FER2013_vs1_f1.pth"  # Đường dẫn tới checkpoint của ResNet50_Attention đã huấn luyện
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 7
TARGET_SIZE = (224, 224)  # Kích thước đầu vào của ResNet50_Attention
BATCH_SIZE = 32

# Label mapping from CK+ to FER-2013 (theo nhãn của ResNet50_Attention)
# CK+ labels: 0=anger, 1=disgust, 2=fear, 3=happiness, 4=sadness, 5=surprise, 6=neutral, 7=contempt
# FER-2013 labels: 0=angry, 1=disgust, 2=fear, 3=happy, 4=neutral, 5=sad, 6=surprise
ck_to_fer_mapping = {
    0: 0,  # anger -> angry
    1: 1,  # disgust -> disgust
    2: 2,  # fear -> fear
    3: 3,  # happiness -> happy
    4: 5,  # sadness -> sad
    5: 6,  # surprise -> surprise
    6: 4,  # neutral -> neutral
    # 7: excluded (contempt)
}

label_to_emotion = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}

# Transformations (theo code mới)
test_transforms = transforms.Compose(
    [
        transforms.Resize(TARGET_SIZE),
        transforms.Grayscale(num_output_channels=3),  # Chuyển thành ảnh xám 3 kênh
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
        # Map CK+ labels to FER-2013 labels
        self.data["emotion"] = self.data["emotion"].map(ck_to_fer_mapping)
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

# Define the ResNet50_Attention model (từ code mới)
class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResNet50_Attention(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50_Attention, self).__init__()
        self.backbone = create_model("resnet50", pretrained=False, features_only=True)
        self.feature_info = self.backbone.feature_info
        self.attention = SEModule(self.feature_info.channels()[3])
        self.bn = nn.BatchNorm2d(self.feature_info.channels()[-1])
        self.dropout = nn.Dropout(0.5)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(self.feature_info.channels()[-1], num_classes))

    def forward(self, x):
        features = self.backbone(x)
        features[3] = self.attention(features[3])
        x = self.avgpool(features[-1])
        x = self.bn(x)
        x = self.dropout(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
# Load the model
model = ResNet50_Attention(num_classes=NUM_CLASSES)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}  # Remove "module." prefix if needed
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
    "ResNet50_FER201_vs1_confusion_matrix_ckextended_raw.png",
    dpi=300,
    bbox_inches="tight",
)
print(
    "Raw confusion matrix saved to 'ResNet50_FER201_vs1_confusion_matrix_ckextended_raw.png'"
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
    "ResNet50_FER201_vs1_confusion_matrix_ckextended_normalized.png",
    dpi=300,
    bbox_inches="tight",
)
print(
    "Normalized confusion matrix saved to 'ResNet50_FER201_vs1_confusion_matrix_ckextended_normalized.png'"
)

# Save both matrices to CSV
cm_raw_df = pd.DataFrame(
    cm_raw,
    index=[label_to_emotion[i] for i in range(NUM_CLASSES)],
    columns=[label_to_emotion[i] for i in range(NUM_CLASSES)],
)
cm_raw_df.to_csv("ResNet50_FER201_vs1_confusion_matrix_ckextended_raw.csv")
print(
    "Raw confusion matrix saved to 'ResNet50_FER201_vs1_confusion_matrix_ckextended_raw.csv'"
)

cm_normalized_df = pd.DataFrame(
    cm_normalized,
    index=[label_to_emotion[i] for i in range(NUM_CLASSES)],
    columns=[label_to_emotion[i] for i in range(NUM_CLASSES)],
)
cm_normalized_df.to_csv(
    "ResNet50_FER201_vs1_confusion_matrix_ckextended_normalized.csv"
)
print(
    "Normalized confusion matrix saved to 'ResNet50_FER201_vs1_confusion_matrix_ckextended_normalized.csv'"
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