import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models.kie.gated_gcn import GatedGCNNet
import configs as cf


class KIEDataset(Dataset):
    def __init__(self, annotation_file, image_dir):
        with open(annotation_file, "r", encoding="utf-8") as f:
            self.annotations = json.load(f)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        boxes = []
        texts = []
        labels = []

        for box in ann["boxes"]:
            boxes.append(box["poly"])
            texts.append(box["text"])
            label_idx = cf.node_labels.index(box["label"])
            labels.append(label_idx)

        return {
            "boxes": np.array(boxes, dtype=np.float32),
            "texts": texts,
            "labels": np.array(labels, dtype=np.int64),
        }


def collate_fn(batch):
    # Implement custom collate function for batching
    all_boxes = []
    all_texts = []
    all_labels = []

    for sample in batch:
        all_boxes.append(torch.tensor(sample["boxes"]))
        all_texts.extend(sample["texts"])
        all_labels.append(torch.tensor(sample["labels"]))

    return {"boxes": all_boxes, "texts": all_texts, "labels": all_labels}


def train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs, device
):
    model.train()

    for epoch in range(num_epochs):
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, batch in enumerate(train_loader):
            boxes = [b.to(device) for b in batch["boxes"]]
            labels = [l.to(device) for l in batch["labels"]]
            texts = batch["texts"]

            # Forward pass
            outputs = model(boxes, texts)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}, Acc: {100.*train_correct/train_total:.2f}%"
                )

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                boxes = [b.to(device) for b in batch["boxes"]]
                labels = [l.to(device) for l in batch["labels"]]
                texts = batch["texts"]

                outputs = model(boxes, texts)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {train_loss/len(train_loader):.4f}, "
            f"Train Acc: {100.*train_correct/train_total:.2f}%, "
            f"Val Loss: {val_loss/len(val_loader):.4f}, "
            f"Val Acc: {100.*val_correct/val_total:.2f}%"
        )

        # Save checkpoint
        checkpoint = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, f"weights/kie/checkpoint_epoch_{epoch+1}.pth")

        model.train()


def main():
    # Initialize model
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
    model = model.to(cf.device)

    # Load datasets
    train_dataset = KIEDataset("data/annotations/train.json", "data/train/images")
    val_dataset = KIEDataset("data/annotations/val.json", "data/val/images")

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn
    )

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=50,
        device=cf.device,
    )


if __name__ == "__main__":
    main()
