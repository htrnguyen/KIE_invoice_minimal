import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from tqdm import tqdm
import logging
from datetime import datetime
from pathlib import Path

from models.kie.gated_gcn import GatedGCNNet
import configs as cf

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)


class KIEDataset(Dataset):
    def __init__(self, annotation_file, max_text_length=50):
        with open(annotation_file, "r", encoding="utf-8") as f:
            self.annotations = json.load(f)
        self.max_text_length = max_text_length

    def __len__(self):
        return len(self.annotations)

    def encode_text(self, text):
        """Encode text thành vector sử dụng alphabet"""
        text = text.upper()
        encoded = []
        for char in text[: self.max_text_length]:
            if char in cf.alphabet:
                encoded.append(cf.alphabet.index(char))
            else:
                encoded.append(cf.alphabet.index(" "))
        # Pad sequence
        while len(encoded) < self.max_text_length:
            encoded.append(0)  # Padding với space
        return encoded

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        boxes = []
        texts = []
        labels = []

        for box in ann["boxes"]:
            # Normalize coordinates to [0,1]
            coords = np.array(box["poly"], dtype=np.float32)
            coords[0::2] = coords[0::2] / ann["width"]  # x coordinates
            coords[1::2] = coords[1::2] / ann["height"]  # y coordinates

            boxes.append(coords)
            texts.append(self.encode_text(box["text"]))
            label_idx = cf.node_labels.index(box["label"])
            labels.append(label_idx)

        return {
            "boxes": np.array(boxes, dtype=np.float32),
            "texts": np.array(texts, dtype=np.int64),
            "labels": np.array(labels, dtype=np.int64),
            "image_id": ann["file_name"],
        }


def collate_fn(batch):
    """Custom collate function để xử lý batches có số lượng boxes khác nhau"""
    all_boxes = []
    all_texts = []
    all_labels = []
    all_image_ids = []

    for sample in batch:
        all_boxes.append(torch.tensor(sample["boxes"]))
        all_texts.append(torch.tensor(sample["texts"]))
        all_labels.append(torch.tensor(sample["labels"]))
        all_image_ids.append(sample["image_id"])

    return {
        "boxes": all_boxes,
        "texts": all_texts,
        "labels": all_labels,
        "image_ids": all_image_ids,
    }


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        learning_rate=0.001,
        weight_decay=1e-4,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.optimizer = Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        self.criterion = torch.nn.CrossEntropyLoss()

        self.best_val_loss = float("inf")
        self.best_val_acc = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Move data to device
            boxes = [b.to(self.device) for b in batch["boxes"]]
            texts = [t.to(self.device) for t in batch["texts"]]
            labels = torch.cat([l.to(self.device) for l in batch["labels"]])

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(boxes, texts)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            total_loss += loss.item()
            pbar.set_postfix(
                {"loss": f"{total_loss/total:.4f}", "acc": f"{100.*correct/total:.2f}%"}
            )

        return total_loss / len(self.train_loader), correct / total

    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Move data to device
                boxes = [b.to(self.device) for b in batch["boxes"]]
                texts = [t.to(self.device) for t in batch["texts"]]
                labels = torch.cat([l.to(self.device) for l in batch["labels"]])

                # Forward pass
                outputs = self.model(boxes, texts)
                loss = self.criterion(outputs, labels)

                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                total_loss += loss.item()

        val_loss = total_loss / len(self.val_loader)
        val_acc = correct / total
        return val_loss, val_acc

    def train(self, num_epochs, checkpoint_dir):
        for epoch in range(num_epochs):
            logging.info(f"\nEpoch {epoch+1}/{num_epochs}")

            # Training
            train_loss, train_acc = self.train_epoch()
            logging.info(
                f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc*100:.2f}%"
            )

            # Validation
            val_loss, val_acc = self.validate()
            logging.info(
                f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc*100:.2f}%"
            )

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(
                    epoch, val_loss, val_acc, checkpoint_dir, is_best=True
                )

            # Regular checkpoint saving
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, val_loss, val_acc, checkpoint_dir)

    def save_checkpoint(self, epoch, val_loss, val_acc, checkpoint_dir, is_best=False):
        checkpoint = {
            "epoch": epoch + 1,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": self.best_val_acc,
        }

        # Save regular checkpoint
        filename = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(checkpoint, filename)
        logging.info(f"Saved checkpoint: {filename}")

        # Save best model
        if is_best:
            best_filename = os.path.join(checkpoint_dir, "model_best.pth")
            torch.save(checkpoint, best_filename)
            logging.info(f"Saved best model: {best_filename}")


def load_pretrained(model, pretrained_path):
    """Load pretrained weights and adapt to current model"""
    if not os.path.exists(pretrained_path):
        logging.warning(f"Pretrained weights not found at {pretrained_path}")
        return model

    logging.info(f"Loading pretrained weights from {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location="cpu")

    # If loading from MC-OCR weights
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Filter out incompatible keys
    model_state_dict = model.state_dict()
    filtered_state_dict = {
        k: v
        for k, v in state_dict.items()
        if k in model_state_dict and v.shape == model_state_dict[k].shape
    }

    # Load weights
    model.load_state_dict(filtered_state_dict, strict=False)
    logging.info(
        f"Loaded {len(filtered_state_dict)}/{len(model_state_dict)} layers from pretrained weights"
    )

    return model


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config
    config = load_config()

    # Create model
    net_params = {
        "in_dim_text": 768,  # LayoutXLM hidden size
        "in_dim_node": 8,  # 8 coordinates for each box
        "in_dim_edge": 2,  # 2D relative position
        "hidden_dim": 256,
        "out_dim": 128,
        "n_classes": len(config["classes"]),
        "dropout": 0.1,
        "L": 5,
        "readout": True,
        "batch_norm": True,
        "residual": True,
        "device": device,
        "in_feat_dropout": 0.1,
    }

    model = GatedGCNNet(net_params).to(device)

    # Load pretrained weights
    pretrained_path = "weights/kie/vi_layoutxlm.pth"
    model = load_pretrained(model, pretrained_path)

    model = model.to(device)

    # Create checkpoint directory
    checkpoint_dir = Path("weights/kie")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_dataset = KIEDataset("data/dataset/annotations/train.json")
    val_dataset = KIEDataset("data/dataset/annotations/val.json")

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=2
    )

    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=2
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=0.0001,  # Lower learning rate for fine-tuning
        weight_decay=1e-4,
    )

    # Start training
    logging.info("Starting training...")
    trainer.train(num_epochs=50, checkpoint_dir=checkpoint_dir)
    logging.info("Training completed!")


if __name__ == "__main__":
    main()
