import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

from scripts.train_kie import KIEDataset, collate_fn
from models.kie.gated_gcn import GatedGCNNet
import configs as cf


def load_model(checkpoint_path):
    """Load trained model from checkpoint"""
    # Initialize model with same parameters
    net_params = {
        "in_dim_text": len(cf.alphabet),
        "in_dim_node": 10,
        "in_dim_edge": 2,
        "hidden_dim": 512,
        "out_dim": 384,
        "n_classes": len(cf.node_labels),
        "in_feat_dropout": 0.1,
        "dropout": 0.0,
        "L": 4,
        "readout": True,
        "graph_norm": True,
        "batch_norm": True,
        "residual": True,
        "device": cf.device,
    }

    model = GatedGCNNet(net_params)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=cf.device)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(cf.device)
    model.eval()

    return model


def evaluate(model, test_loader, device):
    """Evaluate model on test set"""
    all_preds = []
    all_labels = []
    all_image_ids = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move data to device
            boxes = [b.to(device) for b in batch["boxes"]]
            texts = [t.to(device) for t in batch["texts"]]
            labels = torch.cat([l.to(device) for l in batch["labels"]])

            # Forward pass
            outputs = model(boxes, texts)
            _, predicted = outputs.max(1)

            # Collect predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_image_ids.extend(batch["image_ids"])

    return np.array(all_preds), np.array(all_labels), all_image_ids


def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def analyze_errors(predictions, labels, image_ids, label_names):
    """Analyze and log prediction errors"""
    errors = []
    for pred, true, img_id in zip(predictions, labels, image_ids):
        if pred != true:
            errors.append(
                {
                    "image_id": img_id,
                    "true_label": label_names[true],
                    "predicted_label": label_names[pred],
                }
            )

    return errors


def main():
    # Load test dataset
    test_dataset = KIEDataset("data/dataset/annotations/test.json")
    test_loader = DataLoader(
        test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=4
    )

    # Load best model
    model = load_model("weights/kie/model_best.pth")

    # Evaluate
    predictions, labels, image_ids = evaluate(model, test_loader, cf.device)

    # Calculate and print metrics
    report = classification_report(
        labels, predictions, target_names=cf.node_labels, digits=4
    )
    print("\nClassification Report:")
    print(report)

    # Save report to file
    with open("evaluation_report.txt", "w") as f:
        f.write(report)

    # Plot confusion matrix
    plot_confusion_matrix(labels, predictions, cf.node_labels, "confusion_matrix.png")

    # Analyze errors
    errors = analyze_errors(predictions, labels, image_ids, cf.node_labels)

    print("\nError Analysis:")
    print(f"Total errors: {len(errors)}")

    # Save error analysis
    with open("error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    print("\nEvaluation results have been saved to:")
    print("- evaluation_report.txt")
    print("- confusion_matrix.png")
    print("- error_analysis.json")


if __name__ == "__main__":
    main()
