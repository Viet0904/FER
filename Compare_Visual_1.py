import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import shutil

# Đường dẫn đến thư mục Metrics
metrics_folder = "D:\Workspace\CTU\Final_LuanVan\Metrics_DaTinhToan"  # Đường dẫn thực tế

# Đường dẫn để lưu biểu đồ
output_folder = "D:\Workspace\CTU\Final_LuanVan\PhanPhoiAnh\plots"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Kiểm tra thư mục Metrics
if not os.path.exists(metrics_folder):
    raise FileNotFoundError(f"Thư mục {metrics_folder} không tồn tại.")

# Danh sách các mô hình
models = ['ResNet50', 'ResNet152', 'DenseNet121', 'DenseNet169', 'MobileNetV2', 'MobileNetV3',
          'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4']
datasets = ['RAFDB', 'FER2013']
data_splits = ['Train', 'Test', 'Val']
metrics = ['Accuracy', 'Loss', 'F1-score', 'Precision', 'Recall']

# Lưu trữ dữ liệu
data = {dataset: {model: pd.DataFrame() for model in models} for dataset in datasets}

# Đọc các file log.csv chỉ có "vs1"
for filename in os.listdir(metrics_folder):
    if filename.endswith('_metrics_log.csv') and 'vs1' in filename:
        model_dataset = filename.replace('_metrics_log.csv', '')
        for dataset in datasets:
            if dataset in model_dataset:
                for model in models:
                    if model in model_dataset:
                        filepath = os.path.join(metrics_folder, filename)
                        try:
                            df = pd.read_csv(filepath)
                            data[dataset][model] = df
                        except Exception as e:
                            print(f"Lỗi khi đọc file {filename}: {e}")

# Hàm vẽ Line Chart chuẩn hóa cho từng split riêng biệt
def plot_normalized_line(metric, dataset, split, models, data):
    plt.figure(figsize=(18, 10))  # Tăng kích thước biểu đồ để dễ nhìn
    normalized_epochs = np.linspace(0, 100, 100)  # Chuẩn hóa thành 0-100%
    
    for model in models:
        if not data[dataset][model].empty:
            epochs = np.linspace(0, 100, len(data[dataset][model]))
            values = data[dataset][model][f'{split} {metric}']
            f = interp1d(epochs, values, bounds_error=False, fill_value="extrapolate")
            plt.plot(normalized_epochs, f(normalized_epochs), label=model, linewidth=2.5)  # Tăng độ dày đường
    
    plt.xlabel('Normalized Training Progress (%)', fontsize=14)
    plt.ylabel(f'{split} {metric}', fontsize=14)
    plt.title(f'Normalized {split} {metric} Trends - {dataset}', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)  # Tăng kích thước chữ trong legend
    plt.grid(True, linestyle='--', alpha=0.7)  # Thêm lưới để dễ phân biệt
    plt.tight_layout()
    # Lưu trực tiếp mà không hiển thị
    plt.savefig(os.path.join(output_folder, f'{dataset}_{split}_{metric}_normalized_line.png'), bbox_inches='tight')
    plt.close()

# Vẽ và lưu biểu đồ riêng cho từng split của mỗi dataset
for metric in metrics:
    print(f"\nVẽ biểu đồ cho {metric}:")
    for split in data_splits:
        # Vẽ cho RAFDB
        plot_normalized_line(metric, 'RAFDB', split, models, data)
        # Vẽ cho FER2013
        plot_normalized_line(metric, 'FER2013', split, models, data)

# Nén tất cả biểu đồ thành file ZIP
zip_filename = "D:\Workspace\CTU\Final_LuanVan\PhanPhoiAnh\plots.zip"  # Sửa đường dẫn cho phù hợp với local
shutil.make_archive(zip_filename.replace('.zip', ''), 'zip', output_folder)
print(f"Đã nén tất cả biểu đồ vào {zip_filename}")