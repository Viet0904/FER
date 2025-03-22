import os

def hien_thi_cay_thu_muc(duong_dan, tien_to="", muc_do=0):
  """Hiển thị cấu trúc thư mục dạng cây.

  Args:
    duong_dan: Đường dẫn thư mục gốc.
    tien_to: Tiền tố để tạo cấu trúc cây (mục đích để format).
    muc_do: Mức độ sâu của thư mục trong cây.
  """
  try:
    danh_sach = os.listdir(duong_dan)
    so_muc = len(danh_sach)
    for i, ten_muc in enumerate(danh_sach):
      duong_dan_muc = os.path.join(duong_dan, ten_muc)
      if os.path.isdir(duong_dan_muc):
        if i == so_muc - 1:
          print(tien_to + "└── " + ten_muc)
          hien_thi_cay_thu_muc(duong_dan_muc, tien_to + "    ", muc_do + 1)
        else:
          print(tien_to + "├── " + ten_muc)
          hien_thi_cay_thu_muc(duong_dan_muc, tien_to + "│   ", muc_do + 1)
  except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy thư mục '{duong_dan}'.")
  except PermissionError:
    print(f"Lỗi: Không có quyền truy cập thư mục '{duong_dan}'.")
  except Exception as e:
    print(f"Đã xảy ra lỗi: {e}")

# Sử dụng hàm để hiển thị cấu trúc cây thư mục hiện tại
duong_dan_hien_tai = "D:\Workspace\CTU\Final_LuanVan\Log"
hien_thi_cay_thu_muc(duong_dan_hien_tai)

# Ví dụ nếu muốn hiển thị cấu trúc cây của một thư mục khác
# duong_dan_vi_du = "/duong/dan/den/thu/muc"
# hien_thi_cay_thu_muc(duong_dan_vi_du)