import json
import os
import shutil
import random
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split


def load_json_with_encoding(file_path):
    """Load JSON file with proper encoding handling"""
    try:
        # First try UTF-8-BOM
        with open(file_path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except:
        try:
            # Then try regular UTF-8
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return None


def process_annotation(annotation, image_path):
    """Process single annotation"""
    result = {
        "file_name": os.path.basename(image_path),
        "height": annotation.get("h_origin", 0),
        "width": annotation.get("w_origin", 0),
        "boxes": [],
    }

    for cell in annotation.get("cells", []):
        if not isinstance(cell, dict):
            continue

        # Get coordinates
        coords = cell.get("poly", [])
        if len(coords) != 8:
            continue

        # Get text and label
        text = cell.get("vietocr_text", "")
        label = cell.get("cate_text", "OTHER").upper()

        result["boxes"].append({"poly": coords, "text": text, "label": label})

    return result


def analyze_data_distribution(data):
    """Analyze label distribution in the dataset"""
    if not data:
        print("No data to analyze!")
        return {}, {}

    label_counts = defaultdict(int)
    image_per_label = defaultdict(set)

    for img_file, ann in data.items():
        if not isinstance(ann, dict):
            print(f"Warning: Invalid annotation format for {img_file}")
            continue

        cells = ann.get("cells", [])
        if not cells:
            print(f"Warning: No cells found for {img_file}")
            continue

        for cell in cells:
            if not isinstance(cell, dict):
                print(f"Warning: Invalid cell format in {img_file}")
                continue

            if "cate_text" not in cell:
                print(f"Warning: No category text in cell for {img_file}")
                print(f"Cell content: {cell}")
                continue

            label = cell.get("cate_text", "OTHER").upper()
            label_counts[label] += 1
            image_per_label[label].add(img_file)

    if not label_counts:
        print("\nWarning: No valid labels found. This could be because:")
        print("1. The cells array is empty")
        print("2. The cells don't contain 'cate_text' field")
        print("3. The data structure is different than expected")
        return {}, {}

    print("\nLabel distribution in dataset:")
    for label, count in sorted(label_counts.items()):
        print(f"{label}: {count} instances in {len(image_per_label[label])} images")

    return label_counts, image_per_label


def stratified_split(data, image_per_label, split_ratio=(0.7, 0.15, 0.15)):
    """Split data while maintaining label distribution"""
    if not data or not image_per_label:
        print("No data to split!")
        return [], [], []

    # Get all unique images
    all_images = set()
    for images in image_per_label.values():
        all_images.update(images)

    if not all_images:
        print("No images found in the dataset!")
        return [], [], []

    print(f"Total number of images: {len(all_images)}")

    # Create label distribution for each image
    image_labels = defaultdict(list)
    for label, images in image_per_label.items():
        for img in images:
            image_labels[img].append(label)

    # Sort images by their label combination to ensure better distribution
    images_by_labels = defaultdict(list)
    for img, labels in image_labels.items():
        key = tuple(sorted(set(labels)))
        images_by_labels[key].append(img)

    train_images = set()
    val_images = set()
    test_images = set()

    # Split each label combination group maintaining ratios
    for labels, images in images_by_labels.items():
        n_samples = len(images)
        if n_samples < 3:  # If too few samples, put in training
            train_images.update(images)
            continue

        n_train = int(n_samples * split_ratio[0])
        n_val = int(n_samples * split_ratio[1])

        # Shuffle images
        images = list(images)
        random.shuffle(images)

        train_images.update(images[:n_train])
        val_images.update(images[n_train : n_train + n_val])
        test_images.update(images[n_train + n_val :])

    return list(train_images), list(val_images), list(test_images)


def validate_json_structure(data):
    """Validate the structure of loaded JSON data"""
    print("\nValidating JSON structure...")

    if not isinstance(data, dict):
        print(f"Error: Data is not a dictionary, got {type(data)}")
        return False

    # Print first few keys
    print(f"First few image keys: {list(data.keys())[:3]}")

    # Check first item
    first_key = next(iter(data))
    first_item = data[first_key]
    print(
        f"\nFirst item keys: {first_item.keys() if isinstance(first_item, dict) else 'Not a dict'}"
    )

    if not isinstance(first_item, dict):
        print("Error: Annotation items are not dictionaries")
        return False

    # Check cells structure
    if "cells" not in first_item:
        print("Error: No 'cells' field in annotations")
        return False

    cells = first_item["cells"]
    if not isinstance(cells, list):
        print(f"Error: 'cells' is not a list, got {type(cells)}")
        return False

    if cells:
        first_cell = cells[0]
        print(f"\nFirst cell structure: {first_cell}")

        required_fields = ["poly", "vietocr_text", "cate_text"]
        missing_fields = [field for field in required_fields if field not in first_cell]
        if missing_fields:
            print(f"Error: Missing required fields in cells: {missing_fields}")
            return False
    else:
        print("Warning: First item has no cells")

    return True


def prepare_dataset(input_json, image_dir, output_dir, split_ratio=(0.7, 0.15, 0.15)):
    """Prepare dataset by splitting into train/val/test"""
    # Check input paths
    if not os.path.exists(input_json):
        print(f"Error: Annotation file not found: {input_json}")
        return
    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found: {image_dir}")
        return

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

    # Load annotations
    print(f"Loading annotations from {input_json}")
    data = load_json_with_encoding(input_json)
    if not data:
        print("Failed to load annotations!")
        return

    print(f"Loaded {len(data)} image annotations")

    # Validate JSON structure
    if not validate_json_structure(data):
        print("Invalid JSON structure!")
        return

    # Analyze data distribution
    print("\nAnalyzing data distribution...")
    label_counts, image_per_label = analyze_data_distribution(data)

    if not label_counts:
        print("No valid labels found in the dataset!")
        return

    # Split data strategically
    print("\nSplitting dataset...")
    train_files, val_files, test_files = stratified_split(
        data, image_per_label, split_ratio
    )

    splits = {"train": train_files, "val": val_files, "test": test_files}

    # Process and save each split
    for split_name, files in splits.items():
        print(f"\nProcessing {split_name} split ({len(files)} images)...")
        annotations = []
        success_count = 0
        missing_images = []

        for img_file in files:
            # Copy image
            src_img = os.path.join(image_dir, img_file)
            dst_img = os.path.join(output_dir, split_name, "images", img_file)

            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)

                # Process annotation
                if img_file in data:
                    ann = process_annotation(data[img_file], img_file)
                    annotations.append(ann)
                    success_count += 1
            else:
                missing_images.append(img_file)

        if missing_images:
            print(f"Warning: {len(missing_images)} images not found for {split_name}:")
            for img in missing_images[:5]:
                print(f"  - {img}")
            if len(missing_images) > 5:
                print(f"  ... and {len(missing_images) - 5} more")

        print(
            f"Successfully processed {success_count}/{len(files)} images for {split_name}"
        )

        # Save annotations
        output_file = os.path.join(output_dir, "annotations", f"{split_name}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)

    # Verify final distribution
    print("\nVerifying split distributions...")
    for split_name, files in splits.items():
        split_data = {k: data[k] for k in files if k in data}
        label_counts, _ = analyze_data_distribution(split_data)


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    input_json = "data/mcocr_labels.json"
    image_dir = "data/images"
    output_dir = "data/dataset"

    prepare_dataset(input_json, image_dir, output_dir)
