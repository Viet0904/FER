import os
import shutil

source_dir = r'D:\Workspace\CTU\Final_LuanVan\Log'  # Đường dẫn thư mục gốc
destination_dir = r'D:\Workspace\CTU\Final_LuanVan\Metrics_Goc'  # Đường dẫn thư mục đích

# Tạo thư mục đích nếu nó chưa tồn tại
os.makedirs(destination_dir, exist_ok=True)

for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith('_metrics_log.csv'):
            source_file = os.path.join(root, file)
            destination_file = os.path.join(destination_dir, file)
            shutil.copy2(source_file, destination_file)
            print(f"Đã sao chép: {source_file} -> {destination_file}")

print("Hoàn thành!")