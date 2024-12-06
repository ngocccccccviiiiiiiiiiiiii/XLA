import cv2
import matplotlib.pyplot as plt

# Đọc ảnh HR từ file
hr_image_path = 'C:\\Users\\84762\\Desktop\\anh\\path_to_hr_images\\anh3.png'  # Đường dẫn đến ảnh HR
hr_image = cv2.imread(hr_image_path)

# Kiểm tra nếu ảnh không được đọc thành công
if hr_image is None:
    raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {hr_image_path}")

# Chuyển từ BGR sang RGB
hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

# Tạo ảnh LR từ ảnh HR (giảm 2x kích thước)
lr_image = cv2.resize(hr_image, (hr_image.shape[1] // 2, hr_image.shape[0] // 2), interpolation=cv2.INTER_CUBIC)

# Lưu ảnh LR vào file với phần mở rộng đúng (ví dụ: .png)
lr_image_path = 'C:\\Users\\84762\\Desktop\\anh\\path_to_lr_images\\anh3_lr.png'  # Đảm bảo đường dẫn chứa phần mở rộng
lr_image_bgr = cv2.cvtColor(lr_image, cv2.COLOR_RGB2BGR)  # Chuyển từ RGB sang BGR trước khi lưu
cv2.imwrite(lr_image_path, lr_image_bgr)  # Lưu ảnh LR

# Hiển thị ảnh HR và ảnh LR
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Ảnh HR")
plt.imshow(hr_image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Ảnh LR")
plt.imshow(lr_image)
plt.axis('off')

plt.show()

print(f"Đã lưu ảnh LR vào: {lr_image_path}")
