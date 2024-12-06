import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Hàm nâng cấp ảnh LR lên HR
def upgrade_lr_to_hr(model, lr_image_paths):
    for lr_image_path in lr_image_paths:
        # Kiểm tra đường dẫn ảnh LR
        if not os.path.exists(lr_image_path):
            raise ValueError(f"Ảnh LR không tồn tại tại: {lr_image_path}")

        # Đọc ảnh LR
        lr_image = cv2.imread(lr_image_path)

        # Chuyển từ BGR sang RGB
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        # Thay đổi kích thước ảnh
        lr_image = cv2.resize(lr_image, (512, 512))  # Chỉnh kích thước ảnh LR về 512x512

        # Chuẩn hóa ảnh
        lr_image = lr_image.astype('float32') / 255.0

        # Thêm chiều batch
        lr_image = np.expand_dims(lr_image, axis=0)

        # Nâng cấp ảnh LR lên HR
        upgraded_hr_image = model.predict(lr_image)

        # Chuyển kết quả từ tensor thành ảnh
        upgraded_hr_image = np.squeeze(upgraded_hr_image, axis=0)  # Loại bỏ chiều batch
        upgraded_hr_image = (upgraded_hr_image * 255.0).astype('uint8')  # Đưa giá trị pixel về phạm vi [0, 255]

        # Hiển thị ảnh LR và ảnh HR nâng cấp
        plt.figure(figsize=(12, 6))

        # Hiển thị ảnh LR
        plt.subplot(1, 3, 1)
        plt.imshow(lr_image[0])  # Dùng [0] để lấy ảnh từ batch
        plt.title("Ảnh LR")
        plt.axis('off')

        # Hiển thị ảnh HR nâng cấp
        plt.subplot(1, 3, 2)
        plt.imshow(upgraded_hr_image)
        plt.title("Ảnh HR Nâng Cấp")
        plt.axis('off')

        plt.show()

# Đường dẫn ảnh LR
lr_image_paths = [
    'C:\\Users\\84762\\Desktop\\anh\\path_to_lr_images\\anh1_lr.png',
    'C:\\Users\\84762\\Desktop\\anh\\path_to_lr_images\\anh2_lr.png',
    'C:\\Users\\84762\\Desktop\\anh\\path_to_lr_images\\anh3_lr.png'
]  # Đường dẫn đến ảnh LR

# Tải mô hình đã được huấn luyện
model_path = "C:\\Users\\84762\Desktop\\anh\\super_resolution_model.h5"
if not os.path.exists(model_path):
    raise ValueError(f"Mô hình không tồn tại tại: {model_path}")

model = load_model(model_path)

# Nâng cấp ảnh LR lên HR và hiển thị ảnh
upgrade_lr_to_hr(model, lr_image_paths)
