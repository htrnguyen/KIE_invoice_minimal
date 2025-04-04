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

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.kie.gated_gcn import GatedGCNNet
import configs as cf


def load_model(checkpoint_path):
    """Load trained model from checkpoint"""
    # Model parameters
    net_params = {
        "in_dim_text": 768,  # LayoutXLM hidden size
        "in_dim_node": 8,  # 8 coordinates for each box
        "in_dim_edge": 2,  # 2D relative position
        "hidden_dim": 256,
        "out_dim": 128,
        "n_classes": len(cf.node_labels),
        "dropout": 0.1,
        "L": 5,
        "readout": True,
        "batch_norm": True,
        "residual": True,
        "device": "cpu",
        "in_feat_dropout": 0.1,
    }

    # Create model
    model = GatedGCNNet(net_params)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model


def preprocess_image(image_path):
    """Load and preprocess image"""
    image = Image.open(image_path)
    width, height = image.size
    return image, width, height


def predict(model, boxes, texts, device="cpu"):
    """Perform inference on a single image"""
    model = model.to(device)

    with torch.no_grad():
        # Convert inputs to tensors and ensure they are on the correct device
        boxes = [torch.tensor(box, dtype=torch.float32).to(device) for box in boxes]
        texts = [torch.tensor(text, dtype=torch.long).to(device) for text in texts]

        # Forward pass
        outputs = model(boxes, texts)

        # Get predictions
        predictions = torch.argmax(outputs, dim=1)

        # Convert predictions to labels
        labels = []
        for pred in predictions:
            pred_idx = pred.item()  # Convert 0-dim tensor to Python scalar
            if 0 <= pred_idx < len(cf.node_labels):
                labels.append(cf.node_labels[pred_idx])
            else:
                labels.append("UNKNOWN")

    return labels


def visualize_results(image_path, boxes, texts, labels, true_labels=None):
    """Visualize the predicted boxes and labels on the image"""
    # Load image
    img = Image.open(image_path)
    img = np.array(img)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)

    # Define colors for different labels
    colors = {
        "NAME": "red",
        "BRAND": "blue",
        "PRICE": "green",
        "WEIGHT": "purple",
        "WEIGHT_LABEL": "orange",
        "OTHER": "gray",
    }

    # Draw boxes and labels
    for i, (box, text, label) in enumerate(zip(boxes, texts, labels)):
        # Convert box coordinates to polygon format
        x_coords = [box[j] for j in range(0, len(box), 2)]
        y_coords = [box[j] for j in range(1, len(box), 2)]

        # Create polygon
        polygon = patches.Polygon(
            np.column_stack((x_coords, y_coords)),
            linewidth=2,
            edgecolor=colors.get(label, "white"),
            facecolor="none",
            alpha=0.7,
        )
        ax.add_patch(polygon)

        # Add label text
        if isinstance(text, list):
            # If text is encoded, convert back to string
            text_str = ""
            for char_idx in text:
                if char_idx < len(cf.alphabet):
                    text_str += cf.alphabet[char_idx]
        else:
            text_str = text

        # Add label text
        label_text = f"{label}"
        if true_labels and i < len(true_labels):
            label_text += f" (True: {true_labels[i]})"

        ax.text(
            min(x_coords),
            min(y_coords) - 10,
            label_text,
            color=colors.get(label, "white"),
            fontsize=12,
            bbox=dict(facecolor="black", alpha=0.7),
        )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    # Load model
    checkpoint_path = "weights/kie/model_best.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    model = load_model(checkpoint_path)

    # Load a sample image from the dataset
    dataset_path = "data/images"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset directory not found at {dataset_path}")
        return

    # Check if a specific image is requested
    parser = argparse.ArgumentParser(description="Run inference on an image")
    parser.add_argument(
        "--image", type=str, help="Specific image to process (e.g., 0001.jpg)"
    )
    args = parser.parse_args()

    if args.image:
        # Process the specified image
        image_path = os.path.join(dataset_path, args.image)
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return

        print(f"Processing specified image: {image_path}")

        try:
            image, width, height = preprocess_image(image_path)
            print(f"Image size: {width}x{height}")

            # For images without annotations, we need to extract boxes and texts
            # This is a simplified example - in a real application, you would use OCR
            # to extract text and bounding boxes from the image

            # For demonstration, we'll create some example boxes and texts
            # In a real application, you would replace this with actual OCR results
            boxes = []
            texts = []

            # Example: Create a few boxes covering different regions of the image
            # These are just examples - in a real application, you would use OCR
            # to detect text regions
            box1 = [0.1, 0.1, 0.3, 0.1, 0.3, 0.2, 0.1, 0.2]  # Normalized coordinates
            box2 = [0.4, 0.3, 0.6, 0.3, 0.6, 0.4, 0.4, 0.4]
            box3 = [0.2, 0.5, 0.8, 0.5, 0.8, 0.6, 0.2, 0.6]

            # Convert normalized coordinates to pixel coordinates
            box1_pixels = [
                coord * width if i % 2 == 0 else coord * height
                for i, coord in enumerate(box1)
            ]
            box2_pixels = [
                coord * width if i % 2 == 0 else coord * height
                for i, coord in enumerate(box2)
            ]
            box3_pixels = [
                coord * width if i % 2 == 0 else coord * height
                for i, coord in enumerate(box3)
            ]

            boxes.append(box1_pixels)
            boxes.append(box2_pixels)
            boxes.append(box3_pixels)

            # Example texts (in a real application, these would come from OCR)
            # For demonstration, we'll use placeholder texts
            text1 = "EXAMPLE TEXT 1"
            text2 = "EXAMPLE TEXT 2"
            text3 = "EXAMPLE TEXT 3"

            # Encode texts
            def encode_text(text, max_length=50):
                if text is None:
                    return [0] * max_length

                text = text.upper()
                encoded = []
                for char in text[:max_length]:
                    if char in cf.alphabet:
                        encoded.append(cf.alphabet.index(char))
                    else:
                        encoded.append(cf.alphabet.index(" "))
                # Pad sequence
                while len(encoded) < max_length:
                    encoded.append(0)  # Padding with space
                return encoded

            texts.append(encode_text(text1))
            texts.append(encode_text(text2))
            texts.append(encode_text(text3))

            # Perform inference
            labels = predict(model, boxes, texts)

            # Print results
            print("\nPrediction results for image without annotations:")
            print("---------------------------------------------")
            for i, (box, text, label) in enumerate(
                zip(boxes, [text1, text2, text3], labels)
            ):
                print(f"Box {i+1}:")
                print(f"  Coordinates: {box}")
                print(f"  Text: {text}")
                print(f"  Predicted label: {label}")
                print("---------------------------------------------")

            # Visualize results
            visualize_results(image_path, boxes, [text1, text2, text3], labels)

        except Exception as e:
            print(f"Error processing image: {str(e)}")
            import traceback

            traceback.print_exc()

    else:
        # Load annotations first to find a valid image
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
        print(f"Processing image: {image_path}")

        try:
            image, width, height = preprocess_image(image_path)
            print(f"Image size: {width}x{height}")

            # Find annotation for this image
            annotation = next(
                (ann for ann in annotations if ann["file_name"] == valid_image), None
            )

            if annotation is None:
                print(f"Error: No annotation found for image {valid_image}")
                return

            # Extract boxes and texts from annotation
            boxes = []
            texts = []
            true_labels = []
            for box in annotation["boxes"]:
                # Convert normalized coordinates back to pixel coordinates
                coords = np.array(box["poly"], dtype=np.float32)
                coords[0::2] = coords[0::2] * width  # x coordinates
                coords[1::2] = coords[1::2] * height  # y coordinates
                boxes.append(coords.tolist())

                # Encode text
                text = box["text"]
                if text is None:
                    text = ""
                text = text.upper()
                encoded = []
                for char in text[:50]:  # max_text_length = 50
                    if char in cf.alphabet:
                        encoded.append(cf.alphabet.index(char))
                    else:
                        encoded.append(cf.alphabet.index(" "))
                while len(encoded) < 50:
                    encoded.append(0)
                texts.append(encoded)

                # Store true label
                true_labels.append(box["label"])

            # Perform inference
            labels = predict(model, boxes, texts)

            # Print results
            print("\nPrediction results:")
            print("------------------")
            for i, (box, label) in enumerate(zip(boxes, labels)):
                print(f"Box {i+1}:")
                print(f"  Coordinates: {box}")
                print(f"  Text: {annotation['boxes'][i]['text']}")
                print(f"  True label: {annotation['boxes'][i]['label']}")
                print(f"  Predicted label: {label}")
                print("------------------")

            # Visualize results
            visualize_results(
                image_path,
                boxes,
                [box["text"] for box in annotation["boxes"]],
                labels,
                true_labels,
            )

        except Exception as e:
            print(f"Error processing image: {str(e)}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
