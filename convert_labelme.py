import json
import os
from typing import Dict, List, Any

# Mapping từ label sang category_id
CATEGORY_MAPPING = {
    "brand": 1,
    "name": 2,
    "mfg_label": 3,
    "mfg": 4,
    "exp_label": 5,
    "exp": 6,
    "weight_label": 7,
    "weight": 8,
    "other": 0,
}


def round_coordinate(coord: float) -> int:
    """Làm tròn số thực thành số nguyên"""
    return round(coord)


def convert_points_to_poly(points: List[List[float]]) -> List[int]:
    """Chuyển đổi points từ mảng 2 chiều sang mảng 1 chiều và làm tròn tọa độ"""
    return [round_coordinate(coord) for point in points for coord in point]


def convert_labelme_to_target(labelme_data: Dict[str, Any]) -> Dict[str, Any]:
    """Chuyển đổi dữ liệu từ định dạng labelme sang định dạng đích"""
    # Lấy tên file hình từ imagePath
    image_name = os.path.basename(labelme_data["imagePath"])

    # Chuyển đổi shapes thành cells
    cells = []
    for shape in labelme_data["shapes"]:
        cell = {
            "poly": convert_points_to_poly(shape["points"]),
            "cate_id": CATEGORY_MAPPING.get(
                shape["label"], 0
            ),  # Mặc định là 0 nếu không tìm thấy
            "cate_text": shape["label"],
            "vietocr_text": shape["description"],
            "group_id": shape["group_id"],
        }
        cells.append(cell)

    # Tạo cấu trúc dữ liệu đích
    target_data = {image_name: {"cells": cells}}

    return target_data


def process_directory(input_dir: str, output_file: str):
    """Xử lý tất cả file JSON trong thư mục đầu vào"""
    all_data = {}

    # Duyệt qua tất cả file JSON trong thư mục
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_path = os.path.join(input_dir, filename)

            # Đọc file labelme
            with open(input_path, "r", encoding="utf-8") as f:
                labelme_data = json.load(f)

            # Chuyển đổi dữ liệu
            target_data = convert_labelme_to_target(labelme_data)

            # Thêm vào kết quả
            all_data.update(target_data)

    # Lưu kết quả
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)


def main():
    # Đường dẫn thư mục chứa các file labelme
    input_dir = "./images"

    # File output
    output_file = "./mcocr_private_145120aorof.json"

    # Xử lý chuyển đổi
    process_directory(input_dir, output_file)
    print(f"Đã chuyển đổi xong. Kết quả được lưu tại: {output_file}")


if __name__ == "__main__":
    main()
