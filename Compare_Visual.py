import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import shutil

# Đường dẫn đến thư mục Metrics trên Kaggle
metrics_folder = "/kaggle/input/metrics/Metrics"  # Thay bằng đường dẫn thực tế nếu tên dataset khác

# Đường dẫn để lưu biểu đồ
output_folder = "/kaggle/working/plots"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Kiểm tra xem thư mục Metrics có tồn tại không
if not os.path.exists(metrics_folder):
    raise FileNotFoundError(f"Thư mục {metrics_folder} không tồn tại. Vui lòng kiểm tra dataset trên Kaggle.")

# Danh sách các mô hình và tập dữ liệu
models = ['ResNet50', 'ResNet121', 'DenseNet121', 'DenseNet169', 'MobileNetV2', 'MobileNetV3'
          'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 
          'EfficientNetB4', 'ViT', 'Swin']
datasets = ['RAF-DB', 'FER-2013']
data_splits = ['Train', 'Test', 'Val']

# Các metrics cần xử lý
metrics = ['Accuracy', 'Loss', 'F1-score', 'Precision', 'Recall']

# Lưu trữ dữ liệu
data = {dataset: {model: pd.DataFrame() for model in models} for dataset in datasets}

# Đọc tất cả file log.csv
for filename in os.listdir(metrics_folder):
    if filename.endswith('_metrics_log.csv'):
        model_dataset = filename.replace('_metrics_log.csv', '')
        for dataset in datasets:
            if dataset.replace('-', '') in model_dataset:
                for model in models:
                    if model in model_dataset:
                        filepath = os.path.join(metrics_folder, filename)
                        try:
                            df = pd.read_csv(filepath)
                            data[dataset][model] = df
                        except Exception as e:
                            print(f"Lỗi khi đọc file {filename}: {e}")

# Tính giá trị trung bình cho từng metric
avg_metrics = {dataset: {model: {} for model in models} for dataset in datasets}
for dataset in datasets:
    for model in models:
        if not data[dataset][model].empty:
            for split in data_splits:
                for metric in metrics:
                    avg_metrics[dataset][model][f'{split} {metric}'] = round(data[dataset][model][f'{split} {metric}'].mean(), 4)

# Hàm vẽ Bar Chart trung bình
def plot_avg_bar(metric, split, datasets, models, avg_metrics):
    plt.figure(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.35
    raf_values = [avg_metrics['RAF-DB'][m].get(f'{split} {metric}', 0) for m in models]
    fer_values = [avg_metrics['FER-2013'][m].get(f'{split} {metric}', 0) for m in models]
    
    plt.bar(x - width/2, raf_values, width, label='RAF-DB')
    plt.bar(x + width/2, fer_values, width, label='FER-2013')
    plt.xlabel('Models')
    plt.ylabel(f'Average {split} {metric}')
    plt.title(f'Average {split} {metric} Comparison')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{split}_{metric}_avg_bar.png'))
    plt.show()  # Hiển thị biểu đồ
    plt.close()

# Hàm vẽ Line Chart chuẩn hóa
def plot_normalized_line(metric, split, datasets, models, data):
    plt.figure(figsize=(12, 6))
    normalized_epochs = np.linspace(0, 100, 100)
    for dataset in datasets:
        for model in models:
            if not data[dataset][model].empty:
                epochs = np.linspace(0, 100, len(data[dataset][model]))
                values = data[dataset][model][f'{split} {metric}']
                f = interp1d(epochs, values, bounds_error=False, fill_value="extrapolate")
                plt.plot(normalized_epochs, f(normalized_epochs), label=f"{model}_{dataset}")
    
    plt.xlabel('Normalized Training Progress (%)')
    plt.ylabel(f'{split} {metric}')
    plt.title(f'Normalized {split} {metric} Trends')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{split}_{metric}_line.png'))
    plt.show()  # Hiển thị biểu đồ
    plt.close()

# Hàm vẽ Box Plot
def plot_box(metric, split, datasets, models, data):
    plt.figure(figsize=(12, 6))
    all_data = []
    labels = []
    for dataset in datasets:
        for model in models:
            if not data[dataset][model].empty:
                all_data.append(data[dataset][model][f'{split} {metric}'])
                labels.append(f"{model}_{dataset}")
    
    plt.boxplot(all_data, labels=labels)
    plt.xlabel('Models')
    plt.ylabel(f'{split} {metric} Distribution')
    plt.title(f'{split} {metric} Distribution Across Epochs')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{split}_{metric}_box.png'))
    plt.show()  # Hiển thị biểu đồ
    plt.close()

# Hàm vẽ Bar Chart giá trị tốt nhất
def plot_best_bar(metric, split, datasets, models, data):
    plt.figure(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.35
    raf_values = [data['RAF-DB'][m][f'{split} {metric}'].max() if not data['RAF-DB'][m].empty else 0 for m in models]
    fer_values = [data['FER-2013'][m][f'{split} {metric}'].max() if not data['FER-2013'][m].empty else 0 for m in models]
    
    plt.bar(x - width/2, raf_values, width, label='RAF-DB')
    plt.bar(x + width/2, fer_values, width, label='FER-2013')
    plt.xlabel('Models')
    plt.ylabel(f'Best {split} {metric}')
    plt.title(f'Best {split} {metric} Comparison')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{split}_{metric}_best_bar.png'))
    plt.show()  # Hiển thị biểu đồ
    plt.close()

# Vẽ và lưu tất cả các biểu đồ
for metric in metrics:
    print(f"\nVẽ biểu đồ cho {metric}:")
    for split in data_splits:
        plot_avg_bar(metric, split, datasets, models, avg_metrics)
        plot_normalized_line(metric, split, datasets, models, data)
        plot_box(metric, split, datasets, models, data)
        plot_best_bar(metric, split, datasets, models, data)

# Nén tất cả biểu đồ thành file ZIP
zip_filename = "/kaggle/working/plots.zip"
shutil.make_archive(zip_filename.replace('.zip', ''), 'zip', output_folder)
print(f"Đã nén tất cả biểu đồ vào {zip_filename}")

# Xóa thư mục tạm sau khi nén (tùy chọn)
# shutil.rmtree(output_folder)