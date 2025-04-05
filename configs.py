import os
import torch

# Paths
saliency_weight_path = "weights/saliency/u2net.pth"
text_detection_weights_path = "weights/text_detect/craft_mlt_25k.pth"
kie_weight_path = "weights/kie/vi_layoutxlm.pth"

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Saliency detection
saliency_ths = 0.5

# Text detection
craft_config = {
    "text_threshold": 0.7,
    "low_text": 0.4,
    "link_threshold": 0.4,
    "canvas_size": 1024,
    "mag_ratio": 2.0,
    "poly": True,
}

# Text recognition
vietocr_config = {
    "cnn": {
        "pretrained": False,
    },
    "predictor": {
        "beamsearch": False,
    },
}

# KIE
score_ths = 0.5
get_max = True
merge_text = True

# Node labels
node_labels = [
    "NAME",
    "BRAND",
    "MFG_LABEL",
    "MFG",
    "EXP_LABEL",
    "EXP",
    "WEIGHT_LABEL",
    "WEIGHT",
    "OTHER",
]

# Alphabet for text encoding
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 "

img_dir = "./images"
result_img_dir = "./results/model"
raw_img_dir = "./results/raw"
cropped_img_dir = "./results/crop"

infer_batch_vietocr = True  # inference with batch
visualize = False
