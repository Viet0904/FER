import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

class FaceDeblurProcessor:
    def __init__(self, method="simple", blocks=8, factor=3.0):
        """
        Khởi tạo bộ xử lý khử mờ khuôn mặt
        
        Tham số:
            method (str): Phương pháp khử mờ ('simple', 'pixelate', 'deep')
            blocks (int): Số khối cho phương pháp pixelate
            factor (float): Hệ số làm mờ cho phương pháp simple
        """
        self.method = method
        self.blocks = blocks
        self.factor = factor
        
        # Khởi tạo bộ dò khuôn mặt
        self.face_detector = cv2.dnn.readNetFromCaffe(
            "deploy.prototxt.txt",
            "res10_300x300_ssd_iter_140000.caffemodel"
        )
        
        # Khởi tạo mô hình khử mờ nâng cao nếu sử dụng phương pháp deep
        if method == "deep":
            self.deblur_model = self._load_deblur_model()
    
    def _load_deblur_model(self):
        """Tải mô hình khử mờ học sâu"""
        # Đây là một ví dụ đơn giản về mô hình khử mờ
        class DeblurCNN(nn.Module):
            def __init__(self):
                super(DeblurCNN, self).__init__()
                # Encoder
                self.conv1 = nn.Conv2d(3, 64, kernel_size=11, padding=5)
                self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
                self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
                # Decoder
                self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
                self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, padding=2)
                self.deconv3 = nn.ConvTranspose2d(64, 3, kernel_size=11, padding=5)
                
            def forward(self, x):
                # Encoder
                x1 = F.relu(self.conv1(x))
                x2 = F.relu(self.conv2(x1))
                x3 = F.relu(self.conv3(x2))
                # Decoder
                x = F.relu(self.deconv1(x3))
                x = F.relu(self.deconv2(x))
                x = torch.tanh(self.deconv3(x))
                return x
        
        model = DeblurCNN()
        # Trong thực tế, bạn sẽ tải trọng số đã được huấn luyện
        # model.load_state_dict(torch.load("deblur_model.pth"))
        model.eval()
        return model
    
    def detect_faces(self, image, confidence_threshold=0.5):
        """
        Phát hiện khuôn mặt trong ảnh
        
        Tham số:
            image: Ảnh đầu vào (numpy array)
            confidence_threshold: Ngưỡng tin cậy
            
        Trả về:
            List các hộp giới hạn khuôn mặt [(startX, startY, endX, endY),...]
        """
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        face_boxes = []
        
        # Lặp qua các khuôn mặt phát hiện được
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Lọc dựa trên ngưỡng tin cậy
            if confidence > confidence_threshold:
                # Tính toán tọa độ hộp giới hạn
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Đảm bảo hộp giới hạn nằm trong ảnh
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                face_boxes.append((startX, startY, endX, endY))
        
        return face_boxes
    
    def simple_deblur(self, face):
        """
        Khử mờ đơn giản bằng cách tăng độ sắc nét
        
        Tham số:
            face: Vùng khuôn mặt cần khử mờ
            
        Trả về:
            Khuôn mặt đã được khử mờ
        """
        # Chuyển đổi sang grayscale nếu cần
        if len(face.shape) == 3:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        else:
            gray = face
            
        # Áp dụng bộ lọc làm sắc nét (Unsharp Masking)
        gaussian = cv2.GaussianBlur(gray, (0, 0), 3)
        sharpened = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
        
        # Nếu ảnh đầu vào là ảnh màu, chuyển kết quả về màu
        if len(face.shape) == 3:
            # Chuyển đổi ảnh grayscale thành BGR
            sharpened = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        
        return sharpened
    
    def pixelate_deblur(self, face):
        """
        Khử mờ bằng phương pháp pixelate
        
        Tham số:
            face: Vùng khuôn mặt cần khử mờ
            
        Trả về:
            Khuôn mặt đã được khử mờ
        """
        # Tạo bản sao của ảnh để xử lý
        result = face.copy()
        
        # Lấy kích thước ảnh
        (h, w) = face.shape[:2]
        
        # Chia ảnh thành các khối NxN
        xSteps = np.linspace(0, w, self.blocks + 1, dtype="int")
        ySteps = np.linspace(0, h, self.blocks + 1, dtype="int")
        
        # Lặp qua các khối
        for i in range(1, len(ySteps)):
            for j in range(1, len(xSteps)):
                # Tính toán tọa độ bắt đầu và kết thúc cho khối hiện tại
                startX = xSteps[j - 1]
                startY = ySteps[i - 1]
                endX = xSteps[j]
                endY = ySteps[i]
                
                # Trích xuất ROI
                roi = face[startY:endY, startX:endX]
                
                # Tính giá trị trung bình của ROI
                (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
                
                # Vẽ hình chữ nhật với giá trị RGB trung bình
                cv2.rectangle(result, (startX, startY), (endX, endY), (B, G, R), -1)
        
        # Áp dụng bộ lọc làm sắc nét
        kernel = np.array([[-1,-1,-1], 
                           [-1, 9,-1],
                           [-1,-1,-1]])
        sharpened = cv2.filter2D(result, -1, kernel)
        
        return sharpened
    
    def deep_deblur(self, face):
        """
        Khử mờ bằng mô hình học sâu
        
        Tham số:
            face: Vùng khuôn mặt cần khử mờ
            
        Trả về:
            Khuôn mặt đã được khử mờ
        """
        # Chuyển đổi sang tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Chuyển đổi từ numpy sang PIL Image
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        
        # Chuyển đổi sang tensor
        face_tensor = transform(face_pil).unsqueeze(0)
        
        # Khử mờ bằng mô hình học sâu
        with torch.no_grad():
            deblurred_tensor = self.deblur_model(face_tensor)
        
        # Chuyển đổi tensor về numpy array
        deblurred = deblurred_tensor.squeeze(0).permute(1, 2, 0)
        deblurred = ((deblurred * 0.5 + 0.5) * 255).numpy().astype(np.uint8)
        
        # Chuyển đổi từ RGB sang BGR
        deblurred = cv2.cvtColor(deblurred, cv2.COLOR_RGB2BGR)
        
        return deblurred
    
    def deblur_face(self, face):
        """
        Khử mờ khuôn mặt bằng phương pháp đã chọn
        
        Tham số:
            face: Vùng khuôn mặt cần khử mờ
            
        Trả về:
            Khuôn mặt đã được khử mờ
        """
        if self.method == "simple":
            return self.simple_deblur(face)
        elif self.method == "pixelate":
            return self.pixelate_deblur(face)
        elif self.method == "deep":
            return self.deep_deblur(face)
        else:
            raise ValueError(f"Phương pháp không hợp lệ: {self.method}")
    
    def process_image(self, image):
        """
        Xử lý ảnh đầu vào để khử mờ khuôn mặt
        
        Tham số:
            image: Ảnh đầu vào (numpy array)
            
        Trả về:
            Ảnh đã được xử lý
        """
        # Tạo bản sao của ảnh để xử lý
        result = image.copy()
        
        # Phát hiện khuôn mặt
        face_boxes = self.detect_faces(image)
        
        # Xử lý từng khuôn mặt
        for (startX, startY, endX, endY) in face_boxes:
            # Trích xuất vùng khuôn mặt
            face = image[startY:endY, startX:endX]
            
            # Khử mờ khuôn mặt
            deblurred_face = self.deblur_face(face)
            
            # Thay thế vùng khuôn mặt trong ảnh kết quả
            result[startY:endY, startX:endX] = deblurred_face
        
        return result
