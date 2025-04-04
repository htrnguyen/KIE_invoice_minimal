import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path

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
        # Convert inputs to tensors
        boxes = [torch.tensor(box, dtype=torch.float32) for box in boxes]
        texts = [torch.tensor(text, dtype=torch.long) for text in texts]

        # Forward pass
        outputs = model(boxes, texts)
        predictions = torch.argmax(outputs, dim=1)

        # Convert predictions to labels
        labels = [cf.node_labels[pred.item()] for pred in predictions]

    return labels


def main():
    # Load model
    checkpoint_path = "weights/kie/model_best.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    model = load_model(checkpoint_path)

    # Load a sample image from the dataset
    dataset_path = "data/dataset/images"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset directory not found at {dataset_path}")
        return

    # Get list of images
    image_files = [
        f for f in os.listdir(dataset_path) if f.endswith((".jpg", ".png", ".jpeg"))
    ]
    if not image_files:
        print(f"Error: No images found in {dataset_path}")
        return

    # Use the first image
    image_path = os.path.join(dataset_path, image_files[0])
    print(f"Processing image: {image_path}")

    try:
        image, width, height = preprocess_image(image_path)
        print(f"Image size: {width}x{height}")

        # Load corresponding annotation
        annotation_path = "data/dataset/annotations/val.json"
        if not os.path.exists(annotation_path):
            print(f"Error: Annotation file not found at {annotation_path}")
            return

        with open(annotation_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        # Find annotation for this image
        image_id = os.path.basename(image_path)
        annotation = next(
            (ann for ann in annotations if ann["file_name"] == image_id), None
        )

        if annotation is None:
            print(f"Error: No annotation found for image {image_id}")
            return

        # Extract boxes and texts from annotation
        boxes = []
        texts = []
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

    except Exception as e:
        print(f"Error processing image: {str(e)}")


if __name__ == "__main__":
    main()
