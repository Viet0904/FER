# -*- coding: utf-8 -*-
import os
import time
import copy
import random
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# Cấu hình
train_path = "/kaggle/input/raf-db-dataset/DATASET/train"
test_path = "/kaggle/input/raf-db-dataset/DATASET/test"

NUM_CLASSES = 7
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
PATIENCE = 20
USE_WEIGHTED_SAMPLER = True
VAL_SIZE = 0.2
TARGET_SIZE = (224, 224)  # MobileNetV2 hỗ trợ 224x224

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Ánh xạ nhãn số sang chữ (theo nhãn ImageFolder cho RAF-DB: thư mục "1" đến "7")
label_to_emotion = {
    0: "surprise",  # Thư mục "1"
    1: "fear",  # Thư mục "2"
    2: "disgust",  # Thư mục "3"
    3: "happy",  # Thư mục "4"
    4: "sad",  # Thư mục "5"
    5: "angry",  # Thư mục "6"
    6: "neutral",  # Thư mục "7"
}
print("Mapping (ImageFolder label -> emotion):", label_to_emotion)

train_transforms = transforms.Compose(
    [
        transforms.Resize(TARGET_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Grayscale(num_output_channels=3),  # Chuyển tất cả thành ảnh xám
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_val_transforms = transforms.Compose(
    [
        transforms.Resize(TARGET_SIZE),
        transforms.Grayscale(num_output_channels=3),  # Chuyển tất cả thành ảnh xám
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load dataset
train_dataset = datasets.ImageFolder(root=train_path, transform=train_transforms)
test_dataset = datasets.ImageFolder(root=test_path, transform=test_val_transforms)

# Chia tập valid
train_indices, val_indices = train_test_split(
    list(range(len(train_dataset))),
    test_size=VAL_SIZE,
    random_state=SEED,
    stratify=train_dataset.targets,
)

train_split_dataset = copy.deepcopy(train_dataset)
train_split_dataset.samples = [train_dataset.samples[i] for i in train_indices]
train_split_dataset.targets = [train_dataset.targets[i] for i in train_indices]

val_split_dataset = copy.deepcopy(train_dataset)
val_split_dataset.samples = [train_dataset.samples[i] for i in val_indices]
val_split_dataset.transform = test_val_transforms
val_split_dataset.targets = [train_dataset.targets[i] for i in val_indices]


# Hàm trực quan hóa phân bố lớp
def visualize_combined_class_distribution(
    train_dataset, val_dataset, test_dataset, label_to_emotion, save_path=None
):
    # Tính phân bố lớp cho từng tập
    train_class_counts = defaultdict(int)
    val_class_counts = defaultdict(int)
    test_class_counts = defaultdict(int)
    
    for _, label in train_dataset.samples:
        train_class_counts[label_to_emotion[label]] += 1
    for _, label in val_dataset.samples:
        val_class_counts[label_to_emotion[label]] += 1
    for _, label in test_dataset.samples:
        test_class_counts[label_to_emotion[label]] += 1

    # In phân bố lớp
    print("Train class counts:", dict(train_class_counts))
    print("Validation class counts:", dict(val_class_counts))
    print("Test class counts:", dict(test_class_counts))

    # Chuẩn bị dữ liệu để vẽ
    emotions = sorted(train_class_counts.keys())  # Giả sử các nhãn giống nhau giữa các tập
    train_counts = [train_class_counts[emotion] for emotion in emotions]
    val_counts = [val_class_counts[emotion] for emotion in emotions]
    test_counts = [test_class_counts[emotion] for emotion in emotions]

    # Thiết lập biểu đồ
    plt.figure(figsize=(12, 6))
    bar_width = 0.25  # Độ rộng của mỗi thanh
    index = np.arange(len(emotions))  # Vị trí các nhóm thanh

    plt.bar(index, train_counts, bar_width, label="Train", color="skyblue")
    plt.bar(index + bar_width, val_counts, bar_width, label="Validation", color="salmon")
    plt.bar(index + 2 * bar_width, test_counts, bar_width, label="Test", color="lightgreen")

    plt.title("Class Distribution Across Train, Validation, and Test Sets", fontsize=14)
    plt.xlabel("Emotion", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.xticks(index + bar_width, emotions, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    # Lưu ảnh
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved combined class distribution plot to {save_path}")
    plt.close()

# Gọi hàm trực quan hóa kết hợp
visualize_combined_class_distribution(
    train_split_dataset,
    val_split_dataset,
    test_dataset,
    label_to_emotion,
    "combined_class_distribution.png"
)

# Data loaders
if USE_WEIGHTED_SAMPLER:
    class_counts = defaultdict(int)
    for _, label in train_split_dataset.samples:
        class_counts[label] += 1

    class_weights = {i: 1.0 / max(class_counts[i], 1) for i in range(NUM_CLASSES)}
    sample_weights = [class_weights[label] for _, label in train_split_dataset.samples]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_split_dataset), replacement=True
    )
    train_loader = DataLoader(
        train_split_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
else:
    train_loader = DataLoader(
        train_split_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

val_loader = DataLoader(
    val_split_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)


# Xây dựng mô hình MobileNetV2
class MobileNetV2_Model(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2_Model, self).__init__()
        self.backbone = timm.create_model(
            "mobilenetv2_100", pretrained=True, num_classes=num_classes
        )

    def forward(self, x):
        return self.backbone(x)


model = MobileNetV2_Model(num_classes=NUM_CLASSES)
model = model.to(device)
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

# Loss, Optimizer, Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=9, gamma=0.3)


# Hàm tính metrics
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return acc, f1, prec, rec


# Tệp log
metrics_log_file = "MobileNetV2_RAFDB_metrics_log.csv"
confusion_matrix_log_file = "MobileNetV2_RAFDB_confusion_matrix_log.csv"

if os.path.exists(metrics_log_file):
    os.remove(metrics_log_file)
if os.path.exists(confusion_matrix_log_file):
    os.remove(confusion_matrix_log_file)

metrics_header = (
    "Epoch,Test Accuracy,Test Loss,Test F1-score,Test Precision,Test Recall,"
    "Val Accuracy,Val Loss,Val F1-score,Val Precision,Val Recall,"
    "Train Accuracy,Train Loss,Train F1-score,Train Precision,Train Recall\n"
)
with open(metrics_log_file, "w") as f:
    f.write(metrics_header)
with open(confusion_matrix_log_file, "w") as f:
    f.write("Epoch,Confusion Matrix\n")

best_loss = float("inf")
best_f1 = 0.0
best_acc = 0.0
epochs_no_improve = 0

# Training loop
scaler = GradScaler()

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
    print("-" * 30)

    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    start_time = time.time()

    for inputs, labels in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(train_loader.dataset)
    train_acc, train_f1, train_prec, train_rec = compute_metrics(all_labels, all_preds)

    # Validation
    model.eval()
    running_loss_val = 0.0
    all_preds_val = []
    all_labels_val = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            running_loss_val += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds_val.extend(preds.cpu().numpy())
            all_labels_val.extend(labels.cpu().numpy())

    val_loss = running_loss_val / len(val_loader.dataset)
    val_acc, val_f1, val_prec, val_rec = compute_metrics(all_labels_val, all_preds_val)

    # Test
    running_loss_test = 0.0
    all_preds_test = []
    all_labels_test = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            running_loss_test += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds_test.extend(preds.cpu().numpy())
            all_labels_test.extend(labels.cpu().numpy())

    test_loss = running_loss_test / len(test_loader.dataset)
    test_acc, test_f1, test_prec, test_rec = compute_metrics(
        all_preds_test, all_labels_test
    )
    conf_mat = confusion_matrix(all_labels_test, all_preds_test)

    # In kết quả
    epoch_time = time.time() - start_time
    print(
        f"Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | "
        f"Train Prec: {train_prec:.4f} | Train Rec: {train_rec:.4f}"
    )
    print(
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | "
        f"Val Prec: {val_prec:.4f} | Val Rec: {val_rec:.4f}"
    )
    print(
        f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f} | "
        f"Test Prec: {test_prec:.4f} | Test Rec: {test_rec:.4f}"
    )
    print(f"Epoch time: {epoch_time:.2f} sec")

    # Ghi log
    with open(metrics_log_file, "a") as f:
        log_line = (
            f"{epoch},{test_acc:.4f},{test_loss:.4f},{test_f1:.4f},{test_prec:.4f},{test_rec:.4f},"
            f"{val_acc:.4f},{val_loss:.4f},{val_f1:.4f},{val_prec:.4f},{val_rec:.4f},"
            f"{train_acc:.4f},{epoch_loss:.4f},{train_f1:.4f},{train_prec:.4f},{train_rec:.4f}\n"
        )
        f.write(log_line)

    with open(confusion_matrix_log_file, "a") as f:
        f.write(f'{epoch},"{conf_mat.tolist()}"\n')

    scheduler.step()

    # Checkpointing và Early Stopping
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_model_wts_f1 = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
        print("Model improved (F1). Saving best model weights.")
        torch.save(model.state_dict(), "MobileNetV2_RAFDB_f1.pth")
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s).")
        if epochs_no_improve >= PATIENCE:
            print("Early stopping triggered!")
            break

    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts_acc = copy.deepcopy(model.state_dict())
        print("Model improved (Accuracy). Saving best model weights.")
        torch.save(model.state_dict(), "MobileNetV2_RAFDB_acc.pth")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "MobileNetV2_RAFDB_loss.pth")

    torch.cuda.empty_cache()

print("Training complete.")
