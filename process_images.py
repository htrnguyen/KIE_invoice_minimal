import os
import shutil
from PIL import Image


def process_image(image_path):
    """Xử lý ảnh: chỉ giữ lại nửa phía sau"""
    # Mở ảnh
    img = Image.open(image_path)

    # Lấy kích thước ảnh
    width, height = img.size

    # Tính toán điểm giữa
    mid_point = width // 2

    # Cắt ảnh, chỉ giữ lại nửa phía sau
    right_half = img.crop((mid_point, 0, width, height))

    # Tạo ảnh mới với kích thước bằng nửa ảnh gốc
    new_img = Image.new(img.mode, (width // 2, height))

    # Dán nửa phía sau vào ảnh mới
    new_img.paste(right_half, (0, 0))

    # Lưu ảnh mới
    output_path = os.path.join("processed_images", os.path.basename(image_path))
    new_img.save(output_path)
    print(f"Đã xử lý: {image_path}")


def main():
    # Tạo thư mục output nếu chưa tồn tại
    if not os.path.exists("processed_images"):
        os.makedirs("processed_images")

    # Lấy danh sách tất cả ảnh từ thư mục images
    image_dir = "images"
    all_images = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Tính số lượng ảnh cần lấy (một nửa)
    total_images = len(all_images)
    half_count = total_images // 2

    # Copy nửa số ảnh cuối cùng
    for filename in all_images[half_count:]:
        src_path = os.path.join(image_dir, filename)
        dst_path = os.path.join("processed_images", filename)
        shutil.copy2(src_path, dst_path)
        print(f"Đã copy: {filename}")

    print(f"Đã copy {half_count} ảnh cuối từ tổng số {total_images} ảnh!")


if __name__ == "__main__":
    main()
