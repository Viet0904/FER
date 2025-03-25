import os
import pandas as pd
import numpy as np
from pathlib import Path

def calculate_stats(df):
    metrics = ['Test Accuracy', 'Test Loss', 'Test F1-score', 'Test Precision', 'Test Recall',
              'Val Accuracy', 'Val Loss', 'Val F1-score', 'Val Precision', 'Val Recall',
              'Train Accuracy', 'Train Loss', 'Train F1-score', 'Train Precision', 'Train Recall']
    
    means = df[metrics].mean()
    stds = df[metrics].std()
    return means, stds

def process_csv_file(filepath):
    # Đọc file CSV
    df = pd.read_csv(filepath)
    
    # Tính toán mean và std
    means, stds = calculate_stats(df)
    
    # Tạo DataFrame cho mean và std
    mean_row = pd.DataFrame([['Mean'] + means.tolist()], columns=['Epoch'] + means.index.tolist())
    std_row = pd.DataFrame([['Std'] + stds.tolist()], columns=['Epoch'] + stds.index.tolist())
    
    # Thêm vào file gốc
    df_with_stats = pd.concat([df, mean_row, std_row], ignore_index=True)
    df_with_stats.to_csv(filepath, index=False)
    
    return means, stds

def create_summary_tables(folder_path):
    rafdb_data = {}
    fer2013_data = {}
    
    # Các metric cần lấy
    target_metrics = ['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1-score']
    
    # Lặp qua tất cả file trong folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            filepath = os.path.join(folder_path, filename)
            means, stds = process_csv_file(filepath)
            
            # Lấy tên model và dataset
            model_name = filename.replace('_metrics_log.csv', '')
            if 'RAFDB' in filename:
                dataset = 'RAFDB'
                model_name = model_name.replace('RAFDB_', '')
                rafdb_data[model_name] = {
                    metric: f"{means[metric]:.4f} ± {stds[metric]:.4f}" 
                    for metric in target_metrics
                }
            elif 'FER2013' in filename:
                dataset = 'FER2013'
                model_name = model_name.replace('FER2013_', '')
                fer2013_data[model_name] = {
                    metric: f"{means[metric]:.4f} ± {stds[metric]:.4f}" 
                    for metric in target_metrics
                }
    
    # Tạo DataFrame cho từng dataset
    rafdb_df = pd.DataFrame(rafdb_data).T
    fer2013_df = pd.DataFrame(fer2013_data).T
    
    # Đổi tên cột
    column_mapping = {
        'Test Accuracy': 'Accuracy',
        'Test Precision': 'Precision',
        'Test Recall': 'Recall',
        'Test F1-score': 'F1-score'
    }
    rafdb_df = rafdb_df.rename(columns=column_mapping)
    fer2013_df = fer2013_df.rename(columns=column_mapping)
    
    # Sắp xếp theo tên model
    rafdb_df = rafdb_df.sort_index()
    fer2013_df = fer2013_df.sort_index()
    
    # Lưu vào Excel
    with pd.ExcelWriter('model_comparison.xlsx') as writer:
        rafdb_df.to_excel(writer, sheet_name='RAFDB')
        fer2013_df.to_excel(writer, sheet_name='FER2013')

def main():
    folder_path = "D:\Workspace\CTU\Final_LuanVan\Metrics_DaTinhToan"
    if not os.path.exists(folder_path):
        print("Folder không tồn tại!")
        return
    
    create_summary_tables(folder_path)
    print("Đã hoàn thành! Kết quả được lưu trong 'model_comparison.xlsx'")

if __name__ == "__main__":
    main()