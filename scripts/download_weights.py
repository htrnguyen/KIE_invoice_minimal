import os
import gdown
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download


def download_pretrained_weights():
    """Download pretrained weights for models"""
    # Create weights directory if not exists
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)

    # Create subdirectories
    (weights_dir / "text_detect").mkdir(exist_ok=True)
    (weights_dir / "saliency").mkdir(exist_ok=True)
    (weights_dir / "kie").mkdir(exist_ok=True)

    # Download CRAFT text detection weights
    craft_url = "https://drive.google.com/uc?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ"
    craft_path = weights_dir / "text_detect" / "craft_mlt_25k.pth"
    if not craft_path.exists():
        print("Downloading CRAFT weights...")
        gdown.download(craft_url, str(craft_path), quiet=False)

    # Download U2Net saliency weights
    u2net_url = "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ"
    u2net_path = weights_dir / "saliency" / "u2netp.pth"
    if not u2net_path.exists():
        print("Downloading U2Net weights...")
        gdown.download(u2net_url, str(u2net_path), quiet=False)

    # Download LayoutXLM weights from Hugging Face
    print("Downloading LayoutXLM weights...")
    try:
        # Download model weights from Hugging Face
        model_path = hf_hub_download(
            repo_id="microsoft/layoutxlm-base",
            filename="pytorch_model.bin",
            cache_dir=str(weights_dir / "kie"),
        )

        # Copy to our desired location
        target_path = weights_dir / "kie" / "vi_layoutxlm.pth"
        if not target_path.exists():
            import shutil

            shutil.copy2(model_path, target_path)
            print(f"Weights saved to {target_path}")
    except Exception as e:
        print(f"Error downloading from Hugging Face: {str(e)}")
        print("Falling back to backup weights...")
        # Fallback to backup weights from Google Drive
        backup_url = "https://drive.google.com/uc?id=1-kQmvWVhvMrJz0yEKt5rrUGqJ1zB0AqF"
        backup_path = weights_dir / "kie" / "vi_layoutxlm.pth"
        if not backup_path.exists():
            gdown.download(backup_url, str(backup_path), quiet=False)

    print("All weights downloaded successfully!")


if __name__ == "__main__":
    download_pretrained_weights()
