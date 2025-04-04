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
    model = load_model(checkpoint_path)

    # Example usage
    image_path = "path/to/your/image.jpg"  # Replace with your image path
    image, width, height = preprocess_image(image_path)

    # Example boxes and texts (replace with your actual data)
    boxes = [
        [100, 100, 200, 100, 200, 150, 100, 150],  # Example box coordinates
        [300, 200, 400, 200, 400, 250, 300, 250],
    ]

    texts = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]  # Example text encodings

    # Perform inference
    labels = predict(model, boxes, texts)

    # Print results
    for box, label in zip(boxes, labels):
        print(f"Box: {box}")
        print(f"Predicted label: {label}")
        print("---")


if __name__ == "__main__":
    main()
