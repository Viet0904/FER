import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil

# Đường dẫn đến thư mục Metrics
metrics_folder = "D:\Workspace\CTU\Final_LuanVan\Metrics_DaTinhToan"
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

# Danh sách để lưu các file chứa giá trị NaN
files_with_nan = set()

# Đọc các file log.csv chỉ có "vs1" và kiểm tra NaN
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
                            # Kiểm tra NaN trong các cột metric
                            for split in data_splits:
                                for metric in metrics:
                                    column = f'{split} {metric}'
                                    if column in df.columns and df[column].isna().any():
                                        files_with_nan.add(filename)
                                        print(f"File {filename} chứa giá trị NaN trong cột {column}")
                        except Exception as e:
                            print(f"Lỗi khi đọc file {filename}: {e}")

# In danh sách các file chứa giá trị NaN
print("\nDanh sách các file chứa giá trị NaN:")
if files_with_nan:
    for file in sorted(files_with_nan):
        print(file)
else:
    print("Không có file nào chứa giá trị NaN.")

# Hàm vẽ Line Chart không chuẩn hóa
def plot_line_chart(metric, dataset, split, models, data):
    plt.figure(figsize=(18, 10))
    
    # Tìm số epoch lớn nhất để đặt giới hạn trục x
    max_epochs = 0
    for model in models:
        if not data[dataset][model].empty:
            num_epochs = len(data[dataset][model])
            max_epochs = max(max_epochs, num_epochs)

    # Vẽ từng mô hình
    for model in models:
        if not data[dataset][model].empty:
            epochs = np.arange(1, len(data[dataset][model]) + 1)  # Số epoch thực tế (bắt đầu từ 1)
            values = data[dataset][model][f'{split} {metric}']
            
            # Thay NaN bằng 0 để vẽ
            if values.isna().any():
                values = values.fillna(0)
            
            plt.plot(epochs, values, label=model, linewidth=2.5)

    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel(f'{split} {metric}', fontsize=14)
    plt.title(f'{split} {metric} Trends - {dataset}', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(np.arange(0, max_epochs + 1, step=max(1, max_epochs // 10)))
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{dataset}_{split}_{metric}_line.png'), bbox_inches='tight')
    plt.close()

# Vẽ và lưu biểu đồ riêng cho từng split của mỗi dataset
for metric in metrics:
    print(f"\nVẽ biểu đồ cho {metric}:")
    for split in data_splits:
        plot_line_chart(metric, 'RAFDB', split, models, data)
        plot_line_chart(metric, 'FER2013', split, models, data)

# Nén tất cả biểu đồ thành file ZIP
zip_filename = "D:\Workspace\CTU\Final_LuanVan\PhanPhoiAnh\plots.zip"
shutil.make_archive(zip_filename.replace('.zip', ''), 'zip', output_folder)
print(f"Đã nén tất cả biểu đồ vào {zip_filename}")