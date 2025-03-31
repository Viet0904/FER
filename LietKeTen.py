import os


def liet_ke_tap_tin(duong_dan="D:\Workspace\CTU\Final_LuanVan\Metrics_DaTinhToan"):
    """Liệt kê tên các tập tin trong một thư mục.

    Args:
      duong_dan: Đường dẫn đến thư mục.

    Returns:
      Một danh sách các tên tập tin trong thư mục.
    """
    try:
        danh_sach_tap_tin = [
            f
            for f in os.listdir(duong_dan)
            if os.path.isfile(os.path.join(duong_dan, f))
        ]
        return danh_sach_tap_tin
    except FileNotFoundError:
        return f"Lỗi: Không tìm thấy thư mục '{duong_dan}'."
    except Exception as e:
        return f"Lỗi không xác định: {e}"


if __name__ == "__main__":
    folder_path = "D:\Workspace\CTU\Final_LuanVan\Metrics_DaTinhToan"
    cac_tap_tin = liet_ke_tap_tin(folder_path)
    if isinstance(cac_tap_tin, list):
        if cac_tap_tin:
            print(f"Các tập tin trong thư mục '{folder_path}':")
            for tap_tin in cac_tap_tin:
                print(tap_tin)
        else:
            print(f"Thư mục '{folder_path}' không chứa tập tin nào.")
    else:
        print(cac_tap_tin)
