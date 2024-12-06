import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array

# Đọc và chuẩn hóa ảnh HR và LR
def load_and_preprocess_images(hr_image_paths, lr_image_paths, target_size=(512, 512)):
    hr_images = []
    lr_images = []

    for hr_image_path, lr_image_path in zip(hr_image_paths, lr_image_paths):
        # Đọc ảnh HR và LR
        hr_image = cv2.imread(hr_image_path)
        lr_image = cv2.imread(lr_image_path)

        # Kiểm tra ảnh có được đọc thành công không
        if hr_image is None or lr_image is None:
            raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {hr_image_path} hoặc {lr_image_path}")

        # Chuyển từ BGR sang RGB
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        # Thay đổi kích thước ảnh HR và LR về kích thước mới (512x512)
        hr_image = cv2.resize(hr_image, target_size)
        lr_image = cv2.resize(lr_image, target_size)

        # Chuẩn hóa ảnh (chia cho 255 để đưa giá trị vào phạm vi [0, 1])
        hr_image = hr_image.astype('float32') / 255.0
        lr_image = lr_image.astype('float32') / 255.0

        hr_images.append(hr_image)
        lr_images.append(lr_image)

    # Chuyển danh sách thành mảng numpy
    hr_images = np.array(hr_images)
    lr_images = np.array(lr_images)

    return hr_images, lr_images


# Xây dựng mô hình CNN cho Super-Resolution
def build_sr_model(input_shape=(512, 512, 3)):
    model = models.Sequential()

    # Lớp Conv1: 64 filters, kernel 9x9
    model.add(layers.Conv2D(64, (9, 9), activation='relu', padding='same', input_shape=input_shape))
    
    # Lớp Conv2: 32 filters, kernel 1x1
    model.add(layers.Conv2D(32, (1, 1), activation='relu', padding='same'))
    
    # Lớp Conv3: 3 filters, kernel 5x5
    model.add(layers.Conv2D(3, (5, 5), activation='sigmoid', padding='same'))

    # Biến đổi kích thước (Chỉnh lại thành 512x512)
    model.add(layers.UpSampling2D(size=(1, 1)))  # Không thay đổi kích thước (giữ nguyên)

    return model

# Huấn luyện mô hình
def train_model(hr_image_paths, lr_image_paths, epochs=10):
    # Tạo mô hình Super-Resolution
    model = build_sr_model()

    # Biên dịch mô hình
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Chuẩn bị dữ liệu
    hr_images, lr_images = load_and_preprocess_images(hr_image_paths, lr_image_paths)

    # Huấn luyện mô hình
    model.fit(lr_images, hr_images, epochs=epochs, batch_size=1)

    # Lưu mô hình
    model.save('super_resolution_model.h5')
    print("Mô hình đã được lưu với tên super_resolution_model.h5")

    return model

# Dự đoán với mô hình đã huấn luyện
def predict_with_model(model, lr_image_path, target_size=(512, 512)):
    # Kiểm tra đường dẫn ảnh LR
    if not os.path.exists(lr_image_path):
        raise ValueError(f"Ảnh LR không tồn tại tại: {lr_image_path}")

    # Đọc ảnh LR
    lr_image = cv2.imread(lr_image_path)

    # Chuyển từ BGR sang RGB
    lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

    # Thay đổi kích thước ảnh
    lr_image = cv2.resize(lr_image, target_size)

    # Chuẩn hóa ảnh
    lr_image = lr_image.astype('float32') / 255.0

    # Thêm chiều batch
    lr_image = np.expand_dims(lr_image, axis=0)

    # Dự đoán ảnh HR từ LR
    predicted_hr_image = model.predict(lr_image)

    # Chuyển kết quả từ tensor thành ảnh
    predicted_hr_image = np.squeeze(predicted_hr_image, axis=0)  # Loại bỏ chiều batch
    predicted_hr_image = (predicted_hr_image * 255.0).astype('uint8')  # Đưa giá trị pixel về phạm vi [0, 255]

    # Hiển thị ảnh LR và ảnh HR gốc trong cùng một hàng
    plt.figure(figsize=(12, 6))

    # Hiển thị ảnh LR
    plt.subplot(1, 2, 1)
    plt.imshow(lr_image[0])  # Dùng [0] để lấy ảnh từ batch
    plt.title("Ảnh LR")
    plt.axis('off')


    # Hiển thị ảnh HR Nâng Cấp
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_hr_image)
    plt.title("Ảnh HR Dự Đoán Nâng Cấp")
    plt.axis('off')

    plt.show()

# Đường dẫn ảnh HR và LR
hr_image_paths = [
    'C:\\Users\\84762\\Desktop\\anh\\path_to_hr_images\\anh1.png',
    'C:\\Users\\84762\\Desktop\\anh\\path_to_hr_images\\anh2.png',
    'C:\\Users\\84762\\Desktop\\anh\\path_to_hr_images\\anh3.png'
]   # Đường dẫn đến ảnh HR

lr_image_paths = [
    'C:\\Users\\84762\\Desktop\\anh\\path_to_lr_images\\anh1_lr.png',
    'C:\\Users\\84762\\Desktop\\anh\\path_to_lr_images\\anh2_lr.png',
    'C:\\Users\\84762\\Desktop\\anh\\path_to_lr_images\\anh3_lr.png'
] # Đường dẫn đến ảnh LR

# Huấn luyện mô hình
model = train_model(hr_image_paths, lr_image_paths, epochs=10)

# Dự đoán với mô hình đã huấn luyện và hiển thị ảnh
for lr_image_path in lr_image_paths:
    predict_with_model(model, lr_image_path)
