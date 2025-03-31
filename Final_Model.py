import torch
import timm

# Khởi tạo mô hình EfficientNet-B4 với features_only=True để lấy feature maps
model = timm.create_model("efficientnet_b4", pretrained=True, features_only=True)

# Kiểm tra danh sách các layer trong mô hình
print("Danh sách feature map từ các stage:")
for idx, feature in enumerate(model.feature_info.info):
    print(
        f"Stage {idx+1}: Output Channels = {feature['num_chs']}, Stride = {feature['reduction']}"
    )

# Tạo input giả (batch size = 1, 3 kênh màu, 380x380)
x = torch.randn(1, 3, 380, 380)

# Chạy qua model để lấy các feature map
features = model(x)

# In kích thước của từng feature map
for i, f in enumerate(features):
    print(f"Feature map {i}: Shape = {f.shape}")
