import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import cv2
import imageio
import base64
import io
import time
import uuid

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.kie.gated_gcn import GatedGCNNet
from models.saliency.u2net import U2NET
from models.text_detect.craft import CRAFT
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
import configs as cf
from backend.backend_utils import (
    run_ocr,
    make_warp_img,
    resize_and_pad,
    get_group_text_line,
    create_merge_cells,
    get_largest_poly_with_coord,
    get_max_hw,
    get_transform_matrix,
)
from backend.kie.kie_utils import (
    load_gate_gcn_net,
    run_predict,
    postprocess_scores,
    postprocess_write_info,
)


def load_models():
    """Load all required models for the KIE pipeline"""
    # Load saliency model (U2Net)
    saliency_net = U2NET(3, 1)
    saliency_net.load_state_dict(
        torch.load(cf.saliency_weight_path, map_location=torch.device(cf.device))
    )
    saliency_net = saliency_net.to(cf.device)
    saliency_net.eval()

    # Load text detection model (CRAFT)
    text_detector = CRAFT()
    # Load state dict and remove 'module.' prefix if present
    state_dict = torch.load(
        cf.text_detection_weights_path, map_location=torch.device(cf.device)
    )
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v
    text_detector.load_state_dict(new_state_dict)
    text_detector = text_detector.to(cf.device)
    text_detector.eval()

    # Load text recognition model (VietOCR)
    config = Cfg.load_config_from_name("vgg_seq2seq")
    config["cnn"]["pretrained"] = False
    config["device"] = cf.device
    config["predictor"]["beamsearch"] = False
    text_recognizer = Predictor(config)

    # Load KIE model (GatedGCNNet)
    gcn_net = load_gate_gcn_net(cf.device, cf.kie_weight_path)

    return saliency_net, text_detector, text_recognizer, gcn_net


def run_saliency(net, img):
    """Run saliency detection to remove background"""
    # Store original image dimensions
    orig_h, orig_w = img.shape[:2]

    # Resize image for model input
    img_resized = resize_and_pad(img, size=1024, pad=False)

    # Convert to tensor
    img_tensor = (
        torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    )
    img_tensor = img_tensor.to(cf.device)

    # Forward pass
    with torch.no_grad():
        d1 = net(img_tensor)[0]
        pred = d1[:, 0, :, :]
        pred = pred.squeeze().cpu().numpy()

    # Try different thresholds if needed
    thresholds = [cf.saliency_ths, 0.3, 0.2, 0.1]  # Add lower thresholds as fallbacks
    mask = None

    for thresh in thresholds:
        # Threshold
        mask = pred > thresh
        mask = mask.astype(np.uint8)

        # Resize mask back to original image dimensions
        mask = cv2.resize(
            mask.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
        )
        mask = (mask > 0.5).astype(np.uint8)  # Re-threshold after resizing

        # Check if mask is valid (contains some foreground)
        if np.any(mask):
            break

    # If still no valid mask, create a full image mask
    if mask is None or not np.any(mask):
        print("Warning: No valid mask found, using full image mask")
        mask = np.ones((orig_h, orig_w), dtype=np.uint8)

    return mask


def display_results_json(cells, preds, values):
    """Display the results in JSON format"""
    results = []

    for i, (cell, pred, value) in enumerate(zip(cells, preds, values)):
        # Get box coordinates
        poly = cell["poly"]
        x_coords = [float(poly[j]) for j in range(0, len(poly), 2)]
        y_coords = [float(poly[j]) for j in range(1, len(poly), 2)]

        # Get label
        label = cf.node_labels[pred]

        # Get text
        text = cell.get("vietocr_text", "")

        # Create result object
        result = {
            "label": label,
            "text": text,
            "confidence": float(value),
            "bbox": {
                "x_min": float(min(x_coords)),
                "y_min": float(min(y_coords)),
                "x_max": float(max(x_coords)),
                "y_max": float(max(y_coords)),
                "points": [[float(x), float(y)] for x, y in zip(x_coords, y_coords)],
            },
        }

        results.append(result)

    # Print JSON
    print("\nResults in JSON format:")
    print("----------------------")
    print(json.dumps(results, indent=2))

    return results


def process_image(
    image_path, saliency_net, text_detector, text_recognizer, gcn_net, output_path=None
):
    """Process image through the complete KIE pipeline"""
    # Load image
    img = imageio.imread(image_path)
    print(f"Processing image: {image_path}")
    print(f"Image size: {img.shape}")

    # 1. Background subtraction
    print("Step 1: Background subtraction...")
    mask_img = run_saliency(saliency_net, img)
    img[~mask_img.astype(bool)] = 0.0

    # 2. Image alignment
    print("Step 2: Image alignment...")
    warped_img = make_warp_img(img, mask_img)

    # 3 & 4. Text detection and recognition
    print("Step 3 & 4: Text detection and recognition...")
    cells, heatmap, textboxes = run_ocr(
        text_detector, text_recognizer, warped_img, cf.craft_config
    )
    _, lines = get_group_text_line(heatmap, textboxes)
    for line_id, cell in zip(lines, cells):
        cell["group_id"] = line_id

    # Merge adjacent text-boxes
    group_ids = np.array([i["group_id"] for i in cells])
    merged_cells = create_merge_cells(
        text_recognizer, warped_img, cells, group_ids, merge_text=cf.merge_text
    )

    # 5. Key Information Extraction
    print("Step 5: Key Information Extraction...")
    batch_scores, boxes = run_predict(gcn_net, merged_cells, device=cf.device)

    # Post-process scores
    values, preds = postprocess_scores(
        batch_scores, score_ths=cf.score_ths, get_max=cf.get_max
    )
    kie_info = postprocess_write_info(merged_cells, preds)

    # Display results in JSON format
    display_results_json(merged_cells, preds, values)

    # Visualize results
    print("Visualizing results...")
    visualize_results(
        img, warped_img, mask_img, merged_cells, preds, values, boxes, output_path
    )

    return kie_info, img, warped_img, merged_cells, preds, values, boxes


def convert_warped_to_original_coords(warped_point, orig_img_shape, warped_img_shape):
    """
    Convert coordinates from the warped image back to the original image.

    Args:
        warped_point: The points detected in the warped image
        orig_img_shape: Shape of the original image (height, width)
        warped_img_shape: Shape of the warped image (height, width)

    Returns:
        A function that converts coordinates from warped to original image
    """
    # Get the transformation matrix from original to warped
    maxWidth, maxHeight = get_max_hw(warped_point)
    matrix = get_transform_matrix(warped_point, maxWidth, maxHeight)

    # Get the inverse transformation matrix
    inv_matrix = np.linalg.inv(matrix)

    def convert_coords(x, y):
        # Convert from warped to original coordinates
        point = np.array([x, y, 1])
        orig_point = np.dot(inv_matrix, point)
        orig_point = orig_point / orig_point[2]

        # Scale to original image dimensions
        orig_x = int(orig_point[0] * orig_img_shape[1] / warped_img_shape[1])
        orig_y = int(orig_point[1] * orig_img_shape[0] / warped_img_shape[0])

        return orig_x, orig_y

    return convert_coords


def visualize_results(
    orig_img, warped_img, mask_img, cells, preds, values, boxes, output_path=None
):
    """Visualize the predicted boxes and labels on the original image"""
    # Chuyển đổi BGR sang RGB nếu cần
    if len(orig_img.shape) == 3 and orig_img.shape[2] == 3:
        if orig_img[0, 0, 0] > orig_img[0, 0, 2]:  # Nếu kênh Blue > Red
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    # Lấy kích thước ảnh
    h, w = orig_img.shape[:2]

    # Tạo figure với kích thước phù hợp
    fig_width = 12
    fig_height = fig_width * (h / w)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Hiển thị ảnh
    ax.imshow(orig_img)

    # Define colors for different labels
    colors = {
        "NAME": "red",
        "BRAND": "blue",
        "MFG_LABEL": "green",
        "MFG": "green",
        "EXP_LABEL": "purple",
        "EXP": "purple",
        "WEIGHT_LABEL": "orange",
        "WEIGHT": "orange",
        "OTHER": "gray",
    }

    # Draw boxes and labels
    for i, (cell, pred, value) in enumerate(zip(cells, preds, values)):
        # Get box coordinates
        poly = cell["poly"]
        x_coords = [poly[j] for j in range(0, len(poly), 2)]
        y_coords = [poly[j] for j in range(1, len(poly), 2)]

        # Get label
        label = cf.node_labels[pred]
        color = colors.get(label, "white")

        # Create polygon
        polygon = patches.Polygon(
            np.column_stack((x_coords, y_coords)),
            linewidth=2,
            edgecolor=color,
            facecolor="none",
            alpha=0.7,
        )
        ax.add_patch(polygon)

        # Add label text
        text = cell.get("vietocr_text", "")
        label_text = f"{label}: {text} ({value:.2f})"

        # Điều chỉnh vị trí text để không bị che khuất
        text_x = min(x_coords)
        text_y = min(y_coords) - 5

        ax.text(
            text_x,
            text_y,
            label_text,
            color=color,
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor=color, pad=0.5),
            ha="left",
            va="bottom",
        )

    plt.axis("off")
    plt.tight_layout(pad=0)

    # Save figure to file if output_path is provided
    if output_path:
        # Lưu ảnh tạm thời
        temp_path = "temp_visualization.png"
        plt.savefig(temp_path, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()

        # Đọc ảnh đã lưu
        saved_img = cv2.imread(temp_path)

        # Kiểm tra orientation của ảnh gốc
        orig_h, orig_w = orig_img.shape[:2]
        if orig_w > orig_h:  # Nếu ảnh gốc là landscape
            # Xoay ảnh kết quả 90 độ ngược chiều kim đồng hồ
            saved_img = cv2.rotate(saved_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Lưu ảnh đã xoay
        cv2.imwrite(output_path, saved_img)

        # Xóa file tạm
        os.remove(temp_path)

        print(f"Ảnh đã được lưu tại: {output_path}")
    else:
        # Try to display the figure
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display figure: {e}")
            print("Saving to default location instead...")
            default_path = "visualization_result.png"

            # Lưu ảnh tạm thời
            temp_path = "temp_visualization.png"
            plt.savefig(temp_path, dpi=300, bbox_inches="tight", pad_inches=0)
            plt.close()

            # Đọc ảnh đã lưu
            saved_img = cv2.imread(temp_path)

            # Kiểm tra orientation của ảnh gốc
            orig_h, orig_w = orig_img.shape[:2]
            if orig_w > orig_h:  # Nếu ảnh gốc là landscape
                # Xoay ảnh kết quả 90 độ ngược chiều kim đồng hồ
                saved_img = cv2.rotate(saved_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Lưu ảnh đã xoay
            cv2.imwrite(default_path, saved_img)

            # Xóa file tạm
            os.remove(temp_path)

            print(f"Visualization saved to: {default_path}")

    plt.close()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run KIE inference on an image")
    parser.add_argument("--image", type=str, help="Path to the image file")
    parser.add_argument("--output", type=str, help="Output path for visualization")
    args = parser.parse_args()

    # Load models
    print("Loading models...")
    saliency_net, text_detector, text_recognizer, gcn_net = load_models()

    # Process image
    if args.image:
        # Process the specified image
        image_path = args.image
        output_path = args.output if args.output else "visualization_result.png"

        kie_info, orig_img, warped_img, cells, preds, values, boxes = process_image(
            image_path,
            saliency_net,
            text_detector,
            text_recognizer,
            gcn_net,
            output_path,
        )

        # Print extracted information
        print("\nExtracted Information:")
        print("----------------------")
        for key, value in kie_info.items():
            print(f"{key}: {value}")
    else:
        # Find a valid image from the dataset
        dataset_path = "data/images"
        annotation_path = "data/dataset/annotations/val.json"

        if not os.path.exists(annotation_path):
            print(f"Error: Annotation file not found at {annotation_path}")
            return

        with open(annotation_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        # Find first image that exists in both annotations and images directory
        image_files = [
            f for f in os.listdir(dataset_path) if f.endswith((".jpg", ".png", ".jpeg"))
        ]
        valid_image = None
        for image_file in image_files:
            if any(ann["file_name"] == image_file for ann in annotations):
                valid_image = image_file
                break

        if valid_image is None:
            print("Error: No valid images found that have corresponding annotations")
            return

        image_path = os.path.join(dataset_path, valid_image)
        output_path = args.output if args.output else "visualization_result.png"

        kie_info, orig_img, warped_img, cells, preds, values, boxes = process_image(
            image_path,
            saliency_net,
            text_detector,
            text_recognizer,
            gcn_net,
            output_path,
        )

        # Print extracted information
        print("\nExtracted Information:")
        print("----------------------")
        for key, value in kie_info.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
