import os
import shutil

def sao_chep_file_pth(duong_dan_goc, duong_dan_dich_goc):
  """Tìm và sao chép các file .pth vào thư mục tương ứng dựa trên tên file.

  Args:
    duong_dan_goc: Đường dẫn gốc chứa các thư mục mô hình.
    duong_dan_dich_goc: Đường dẫn gốc nơi các thư mục RAFDB và FER2013 sẽ được tạo.
  """
  try:
    for thu_muc in os.listdir(duong_dan_goc):
      duong_dan_thu_muc = os.path.join(duong_dan_goc, thu_muc)
      if os.path.isdir(duong_dan_thu_muc):
        for ten_file in os.listdir(duong_dan_thu_muc):
          if ten_file.endswith(".pth"):
            if "RAFDB" in ten_file:
              duong_dan_dich = os.path.join(duong_dan_dich_goc, "RAFDB")
              if not os.path.exists(duong_dan_dich):
                os.makedirs(duong_dan_dich)
              duong_dan_file_goc = os.path.join(duong_dan_thu_muc, ten_file)
              duong_dan_file_dich = os.path.join(duong_dan_dich, ten_file)
              shutil.copy2(duong_dan_file_goc, duong_dan_file_dich)
              print(f"Đã sao chép '{ten_file}' vào thư mục RAFDB.")
            elif "FER2013" in ten_file:
              duong_dan_dich = os.path.join(duong_dan_dich_goc, "FER2013")
              if not os.path.exists(duong_dan_dich):
                os.makedirs(duong_dan_dich)
              duong_dan_file_goc = os.path.join(duong_dan_thu_muc, ten_file)
              duong_dan_file_dich = os.path.join(duong_dan_dich, ten_file)
              shutil.copy2(duong_dan_file_goc, duong_dan_file_dich)
              print(f"Đã sao chép '{ten_file}' vào thư mục FER2013.")
  except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy thư mục gốc '{duong_dan_goc}' hoặc thư mục đích '{duong_dan_dich_goc}'.")
  except PermissionError:
    print(f"Lỗi: Không có quyền truy cập vào thư mục gốc '{duong_dan_goc}' hoặc thư mục đích '{duong_dan_dich_goc}'.")
  except Exception as e:
    print(f"Đã xảy ra lỗi: {e}")

# Sử dụng hàm với đường dẫn gốc và đường dẫn đích của bạn
duong_dan_goc = r"D:\Workspace\CTU\Final_LuanVan\Log"  # Đường dẫn gốc chứa các thư mục mô hình
duong_dan_dich_goc = r"D:\Workspace\CTU\Final_LuanVan\Model" # Đường dẫn đích

sao_chep_file_pth(duong_dan_goc, duong_dan_dich_goc)