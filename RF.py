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
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Cấu hình
train_path = "/kaggle/input/raf-db-dataset/DATASET/train"  # Điều chỉnh đường dẫn phù hợp với dữ liệu của bạn
test_path = "/kaggle/input/raf-db-dataset/DATASET/test"    # Điều chỉnh đường dẫn phù hợp với dữ liệu của bạn

NUM_CLASSES = 7
VAL_SIZE = 0.2
TARGET_SIZE = (224, 224)  # Kích thước ảnh đầu vào
USE_GRAYSCALE = True      # Chuyển sang ảnh xám để giảm kích thước vector đặc trưng

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Ánh xạ nhãn số sang chữ (theo nhãn ImageFolder cho RAF-DB)
label_to_emotion = {
    0: "surprise",  # Thư mục "1"
    1: "fear",      # Thư mục "2"
    2: "disgust",   # Thư mục "3"
    3: "happy",     # Thư mục "4"
    4: "sad",       # Thư mục "5"
    5: "angry",     # Thư mục "6"
    6: "neutral",   # Thư mục "7"
}
print("Mapping (ImageFolder label -> emotion):", label_to_emotion)

# Transforms cho tiền xử lý ảnh
if USE_GRAYSCALE:
    train_transforms = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.Grayscale(num_output_channels=1),  # Ảnh xám 1 kênh
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Chuẩn hóa cho ảnh xám
    ])
    
    test_val_transforms = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.Grayscale(num_output_channels=1),  # Ảnh xám 1 kênh
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Chuẩn hóa cho ảnh xám
    ])
else:
    train_transforms = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_val_transforms = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Load datasets
train_dataset = datasets.ImageFolder(root=train_path, transform=train_transforms)
test_dataset = datasets.ImageFolder(root=test_path, transform=test_val_transforms)

# Chia tập huấn luyện và validation
train_indices, val_indices = train_test_split(
    list(range(len(train_dataset))),
    test_size=VAL_SIZE,
    random_state=SEED,
    stratify=train_dataset.targets
)

# Tạo dataset splits
train_split_dataset = copy.deepcopy(train_dataset)
train_split_dataset.samples = [train_dataset.samples[i] for i in train_indices]
train_split_dataset.targets = [train_dataset.targets[i] for i in train_indices]

val_split_dataset = copy.deepcopy(train_dataset)
val_split_dataset.samples = [train_dataset.samples[i] for i in val_indices]
val_split_dataset.transform = test_val_transforms
val_split_dataset.targets = [train_dataset.targets[i] for i in val_indices]

# Tạo dataloaders
train_loader = DataLoader(train_split_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_split_dataset, batch_size=64, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

# Hàm chuyển đổi ảnh từ tensor sang mảng 1D
def convert_images_to_1d(dataloader):
    images = []
    labels = []
    for imgs, lbls in dataloader:
        for img, lbl in zip(imgs, lbls):
            # Chuyển tensor thành numpy array và làm phẳng
            flat_img = img.numpy().flatten()
            images.append(flat_img)
            labels.append(lbl.item())
    
    return np.array(images), np.array(labels)

# Chuyển đổi dữ liệu
print("Chuyển đổi dữ liệu sang biểu diễn 1D...")
X_train, y_train = convert_images_to_1d(train_loader)
X_val, y_val = convert_images_to_1d(val_loader)
X_test, y_test = convert_images_to_1d(test_loader)

print(f"Kích thước X_train: {X_train.shape}")
print(f"Kích thước X_val: {X_val.shape}")
print(f"Kích thước X_test: {X_test.shape}")

# Giảm chiều dữ liệu (tùy chọn)
# Nếu vector đặc trưng quá lớn, bạn có thể sử dụng PCA để giảm chiều
from sklearn.decomposition import PCA

# Áp dụng PCA nếu số chiều > 1000
if X_train.shape[1] > 1000:
    n_components = min(1000, X_train.shape[0], X_train.shape[1])
    print(f"Áp dụng PCA để giảm chiều xuống {n_components} chiều...")
    pca = PCA(n_components=n_components, random_state=SEED)
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)
    print(f"Kích thước X_train sau PCA: {X_train.shape}")

# Hàm đánh giá và hiển thị kết quả
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\nKết quả của {model_name}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    
    # Vẽ confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()
    
    emotion_names = [label_to_emotion[i] for i in range(NUM_CLASSES)]
    tick_marks = np.arange(len(emotion_names))
    plt.xticks(tick_marks, emotion_names, rotation=45)
    plt.yticks(tick_marks, emotion_names)
    
    # Hiển thị giá trị trong ma trận
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {'acc': acc, 'f1': f1, 'precision': prec, 'recall': rec}

# Huấn luyện và đánh giá Random Forest
print("\nHuấn luyện Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_val_acc = rf_model.score(X_val, y_val)
print(f"Validation accuracy của Random Forest: {rf_val_acc:.4f}")
rf_metrics = evaluate_model(rf_model, X_test, y_test, 'Random Forest')

# Huấn luyện và đánh giá SVM
print("\nHuấn luyện SVM...")
svm_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=SEED)
svm_model.fit(X_train, y_train)
svm_val_acc = svm_model.score(X_val, y_val)
print(f"Validation accuracy của SVM: {svm_val_acc:.4f}")
svm_metrics = evaluate_model(svm_model, X_test, y_test, 'SVM')

# Huấn luyện và đánh giá Gradient Boosting
print("\nHuấn luyện Gradient Boosting...")
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=SEED)
gb_model.fit(X_train, y_train)
gb_val_acc = gb_model.score(X_val, y_val)
print(f"Validation accuracy của Gradient Boosting: {gb_val_acc:.4f}")
gb_metrics = evaluate_model(gb_model, X_test, y_test, 'Gradient Boosting')

# So sánh các mô hình
models = ['Random Forest', 'SVM', 'Gradient Boosting']
accuracies = [rf_metrics['acc'], svm_metrics['acc'], gb_metrics['acc']]
f1_scores = [rf_metrics['f1'], svm_metrics['f1'], gb_metrics['f1']]

plt.figure(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(models))

plt.bar(index, accuracies, bar_width, label='Accuracy', color='skyblue')
plt.bar(index + bar_width, f1_scores, bar_width, label='F1 Score', color='salmon')

plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Comparison of Models')
plt.xticks(index + bar_width/2, models)
plt.legend()
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nSo sánh các mô hình:")
print(f"{'Model':<20} {'Accuracy':<10} {'F1 Score':<10} {'Precision':<10} {'Recall':<10}")
print("-" * 60)
print(f"{'Random Forest':<20} {rf_metrics['acc']:<10.4f} {rf_metrics['f1']:<10.4f} {rf_metrics['precision']:<10.4f} {rf_metrics['recall']:<10.4f}")
print(f"{'SVM':<20} {svm_metrics['acc']:<10.4f} {svm_metrics['f1']:<10.4f} {svm_metrics['precision']:<10.4f} {svm_metrics['recall']:<10.4f}")
print(f"{'Gradient Boosting':<20} {gb_metrics['acc']:<10.4f} {gb_metrics['f1']:<10.4f} {gb_metrics['precision']:<10.4f} {gb_metrics['recall']:<10.4f}")

# Lưu mô hình tốt nhất
best_model_name = models[np.argmax(accuracies)]
print(f"\nMô hình tốt nhất dựa trên accuracy: {best_model_name}")

best_model = None
if best_model_name == 'Random Forest':
    best_model = rf_model
elif best_model_name == 'SVM':
    best_model = svm_model
else:
    best_model = gb_model

import joblib
joblib.dump(best_model, 'best_model.pkl')
print("Đã lưu mô hình tốt nhất thành 'best_model.pkl'")
