import json
import os
from typing import Dict, List, Any
from PIL import Image

# Mapping từ label sang category_id
CATEGORY_MAPPING = {
    "brand": 1,  # Thương hiệu sản phẩm
    "name": 2,  # Tên sản phẩm
    "mfg_label": 3,  # Nhãn NSX
    "mfg": 4,  # Ngày sản xuất
    "exp_label": 5,  # Nhãn HSD
    "exp": 6,  # Hạn sử dụng
    "weight_label": 7,  # Nhãn khối lượng
    "weight": 8,  # Giá trị khối lượng
    "other": 0,  # Thông tin khác
}

# Mapping từ label sang group_id theo quy tắc
GROUP_MAPPING = {
    "brand": 0,  # Group 0: Thông tin độc lập
    "name": 0,  # Group 0: Thông tin độc lập
    "mfg_label": 1,  # Group 1: Thông tin ngày sản xuất
    "mfg": 1,  # Group 1: Thông tin ngày sản xuất
    "exp_label": 2,  # Group 2: Thông tin hạn sử dụng
    "exp": 2,  # Group 2: Thông tin hạn sử dụng
    "weight_label": 3,  # Group 3: Thông tin khối lượng
    "weight": 3,  # Group 3: Thông tin khối lượng
    "other": 0,  # Group 0: Thông tin khác
}


def round_coordinate(coord: float) -> int:
    """Làm tròn số thực thành số nguyên"""
    return round(coord)


def convert_points_to_poly(points: List[List[float]]) -> List[int]:
    """Chuyển đổi points từ mảng 2 chiều sang mảng 1 chiều và làm tròn tọa độ"""
    return [round_coordinate(coord) for point in points for coord in point]


def get_group_id(label: str) -> int:
    """Tự động lấy group_id dựa trên label"""
    return GROUP_MAPPING.get(label, 0)


def get_image_dimensions(image_path: str, image_dir: str) -> tuple:
    """Lấy kích thước ảnh gốc"""
    # Lấy tên file từ đường dẫn đầy đủ
    image_name = os.path.basename(image_path)
    # Tạo đường dẫn mới đến thư mục chứa ảnh
    new_image_path = os.path.join(image_dir, image_name)

    with Image.open(new_image_path) as img:
        return img.size


def convert_labelme_to_target(
    labelme_data: Dict[str, Any], image_dir: str
) -> Dict[str, Any]:
    """Chuyển đổi dữ liệu từ định dạng labelme sang định dạng đích"""
    # Lấy tên file hình từ imagePath
    image_name = os.path.basename(labelme_data["imagePath"])

    # Lấy kích thước ảnh gốc
    w_origin, h_origin = get_image_dimensions(labelme_data["imagePath"], image_dir)

    # Chuyển đổi shapes thành cells
    cells = []
    for shape in labelme_data["shapes"]:
        label = shape["label"]
        cell = {
            "poly": convert_points_to_poly(shape["points"]),
            "cate_id": CATEGORY_MAPPING.get(label, 0),
            "cate_text": label,
            "vietocr_text": shape.get("description", ""),
            "group_id": get_group_id(label),  # Tự động gán group_id
        }
        cells.append(cell)

    # Tạo cấu trúc dữ liệu đích
    target_data = {
        image_name: {"cells": cells, "h_origin": h_origin, "w_origin": w_origin}
    }

    return target_data


def process_directory(input_dir: str, output_file: str, image_dir: str):
    """Xử lý tất cả file JSON trong thư mục đầu vào"""
    # Dictionary để lưu tất cả dữ liệu đã xử lý
    all_data = {}

    # Duyệt qua tất cả file JSON trong thư mục
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_path = os.path.join(input_dir, filename)

            # Đọc file labelme
            with open(input_path, "r", encoding="utf-8") as f:
                labelme_data = json.load(f)

            # Chuyển đổi dữ liệu
            target_data = convert_labelme_to_target(labelme_data, image_dir)

            # Thêm vào kết quả
            all_data.update(target_data)

            print(f"Đã xử lý file: {filename}")

    # Lưu tất cả kết quả vào một file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)


def main():
    # Đường dẫn thư mục chứa các file labelme
    input_dir = "./label"

    # Đường dẫn thư mục chứa ảnh
    image_dir = "./images"

    # File output
    output_file = "./preprocessed_data/processed_labels.json"

    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Xử lý chuyển đổi
    process_directory(input_dir, output_file, image_dir)
    print(f"Đã chuyển đổi xong. Kết quả được lưu tại: {output_file}")


if __name__ == "__main__":
    main()
