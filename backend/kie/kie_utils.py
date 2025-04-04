import copy
import imageio
import os

import cv2
import numpy as np
import torch
import dgl

import configs as cf
from models.kie.gated_gcn import GatedGCNNet
from backend.backend_utils import timer


def load_gate_gcn_net(device, weight_path):
    """Load GatedGCNNet model from weights"""
    # Create model with default parameters
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
        "device": device,
        "in_feat_dropout": 0.1,
    }

    model = GatedGCNNet(net_params)
    model = model.to(device)

    # Load weights
    if os.path.exists(weight_path):
        checkpoint = torch.load(weight_path, map_location=device)

        # Filter out incompatible keys
        model_state_dict = model.state_dict()
        filtered_state_dict = {}

        for k, v in checkpoint.items():
            # Skip LayoutLM related keys
            if k.startswith("layoutlmv2.") or k.startswith("layoutxlm."):
                continue

            # Remove 'module.' prefix if present
            if k.startswith("module."):
                k = k[7:]

            if k in model_state_dict and v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v

        # Load filtered weights
        model.load_state_dict(filtered_state_dict, strict=False)
        print(
            f"Loaded {len(filtered_state_dict)}/{len(model_state_dict)} layers from pretrained weights"
        )
    else:
        print(f"Warning: Weight file not found at {weight_path}")

    return model


def make_text_encode(text):
    text_encode = []
    for t in text.upper():
        if t not in cf.alphabet:
            text_encode.append(cf.alphabet.index(" "))
        else:
            text_encode.append(cf.alphabet.index(t))
    return np.array(text_encode)


def prepare_data(cells, text_key="vietocr_text"):
    texts = []
    text_lengths = []
    polys = []
    for cell in cells:
        text = cell[text_key]
        text_encode = make_text_encode(text)
        text_lengths.append(text_encode.shape[0])
        texts.append(text_encode)

        poly = copy.deepcopy(cell["poly"].tolist())
        poly.append(np.max(poly[0::2]) - np.min(poly[0::2]))
        poly.append(np.max(poly[1::2]) - np.min(poly[1::2]))
        poly = list(map(int, poly))
        polys.append(poly)

    texts = np.array(texts, dtype=object)
    text_lengths = np.array(text_lengths)
    polys = np.array(polys)
    return texts, text_lengths, polys


def prepare_pipeline(boxes, edge_data, text, text_length):
    box_min = boxes.min(0)
    box_max = boxes.max(0)

    boxes = (boxes - box_min) / (box_max - box_min)
    boxes = (boxes - 0.5) / 0.5

    edge_min = edge_data.min(0)
    edge_max = edge_data.max(0)

    edge_data = (edge_data - edge_min) / (edge_max - edge_min)
    edge_data = (edge_data - 0.5) / 0.5

    return boxes, edge_data, text, text_length


@timer
def prepare_graph(cells):
    texts, text_lengths, boxes = prepare_data(cells)

    origin_boxes = boxes.copy()
    node_nums = text_lengths.shape[0]

    src = []
    dst = []
    edge_data = []
    for i in range(node_nums):
        for j in range(node_nums):
            if i == j:
                continue

            edata = []
            # y distance
            y_distance = np.mean(boxes[i][:8][1::2]) - np.mean(boxes[j][:8][1::2])
            # w = boxes[i, 8]
            h = boxes[i, 9]
            if np.abs(y_distance) > 3 * h:
                continue

            x_distance = np.mean(boxes[i][:8][0::2]) - np.mean(boxes[j][:8][0::2])
            edata.append(y_distance)
            edata.append(x_distance)

            edge_data.append(edata)
            src.append(i)
            dst.append(j)

    edge_data = np.array(edge_data)
    g = dgl.DGLGraph()
    g.add_nodes(node_nums)
    g.add_edges(src, dst)

    boxes, edge_data, text, text_length = prepare_pipeline(
        boxes, edge_data, texts, text_lengths
    )
    boxes = torch.from_numpy(boxes).float()
    edge_data = torch.from_numpy(edge_data).float()

    tab_sizes_n = g.number_of_nodes()
    tab_snorm_n = torch.FloatTensor(tab_sizes_n, 1).fill_(1.0 / float(tab_sizes_n))
    snorm_n = tab_snorm_n.sqrt()

    tab_sizes_e = g.number_of_edges()
    tab_snorm_e = torch.FloatTensor(tab_sizes_e, 1).fill_(1.0 / float(tab_sizes_e))
    snorm_e = tab_snorm_e.sqrt()

    max_length = text_lengths.max()
    new_text = [
        np.expand_dims(np.pad(t, (0, max_length - t.shape[0]), "constant"), axis=0)
        for t in text
    ]
    texts = np.concatenate(new_text)

    texts = torch.from_numpy(np.array(texts))
    text_length = torch.from_numpy(np.array(text_length))

    graph_node_size = [g.number_of_nodes()]
    graph_edge_size = [g.number_of_edges()]

    return (
        g,
        boxes,
        edge_data,
        snorm_n,
        snorm_e,
        texts,
        text_length,
        origin_boxes,
        graph_node_size,
        graph_edge_size,
    )


@timer
def run_predict(gcn_net, merged_cells, device="cpu"):
    # Force CPU for DGL operations
    dgl_device = "cpu"

    # Prepare data for the model
    # The model expects a list of tensors for each batch
    # We'll create a single batch with all cells
    batch_boxes = []
    batch_texts = []

    # Get image dimensions for normalization
    all_x = []
    all_y = []
    for cell in merged_cells:
        poly = cell.get("poly", [])
        if isinstance(poly, np.ndarray):
            poly = poly.tolist()
        x_coords = poly[0::2]
        y_coords = poly[1::2]
        all_x.extend(x_coords)
        all_y.extend(y_coords)

    # Get image bounds
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    # Process each cell
    for cell in merged_cells:
        # Extract text and box data
        text = cell.get("vietocr_text", "")
        poly = cell.get("poly", [])

        if isinstance(poly, np.ndarray):
            poly = poly.tolist()

        # Normalize coordinates to 0-1000 range
        normalized_poly = []
        for i in range(0, len(poly), 2):
            # Normalize x coordinate
            x = int((poly[i] - x_min) * 1000 / (x_max - x_min))
            # Normalize y coordinate
            y = int((poly[i + 1] - y_min) * 1000 / (y_max - y_min))
            normalized_poly.extend([x, y])

        # Convert text to tensor
        text_tensor = torch.tensor([ord(c) for c in text], dtype=torch.long)

        # Convert box to tensor
        box_tensor = torch.tensor(normalized_poly, dtype=torch.float32)

        batch_texts.append(text_tensor)
        batch_boxes.append(box_tensor)

    # Stack all boxes into a single tensor for the batch
    batch_boxes = torch.stack(batch_boxes) if batch_boxes else torch.empty(0, 8)

    # Create a single batch
    boxes = [batch_boxes]  # List with a single tensor containing all boxes
    texts = [batch_texts]  # List of lists of tensors

    # Call the forward method with the correct arguments
    batch_scores = gcn_net.forward(boxes, texts)

    # Return the original boxes for visualization
    return batch_scores, [
        b.numpy() if isinstance(b, torch.Tensor) else b for b in batch_boxes
    ]


@timer
def postprocess_scores(batch_scores, score_ths=0.98, get_max=False):
    values, preds = [], []
    batch_scores = batch_scores.cpu().softmax(1)
    for score in batch_scores:
        _score = score.detach().cpu().numpy()
        values.append(_score.max())
        pred_index = np.argmax(_score)
        if get_max:
            preds.append(pred_index)
        else:
            if pred_index != 0 and _score.max() >= score_ths:
                preds.append(pred_index)
            else:
                preds.append(0)

    preds = np.array(preds)
    return values, preds


@timer
def postprocess_write_info(merged_cells, preds, text_key="vietocr_text"):
    kie_info = dict()
    preds = np.array(preds)

    # Process each label type
    for i in range(1, len(cf.node_labels)):  # Skip 'OTHER' class
        indexes = np.where(preds == i)[0]
        if len(indexes) > 0:
            # For label fields (MFG_LABEL, EXP_LABEL, WEIGHT_LABEL), we want to keep them separate
            if cf.node_labels[i].endswith("_LABEL"):
                text_output = " ".join(
                    merged_cells[index][text_key] for index in indexes
                )
                kie_info[cf.node_labels[i]] = text_output
            else:
                # For value fields (MFG, EXP, WEIGHT), we want to concatenate them
                text_output = " ".join(
                    merged_cells[index][text_key] for index in indexes
                )
                kie_info[cf.node_labels[i]] = text_output

    return kie_info


@timer
def vis_kie_pred(img, preds, values, boxes, save_path):
    vis_img = img.copy()
    length = preds.shape[0]
    for i in range(length):

        pred_id = preds[i]
        if pred_id != 0:
            msg = "{}-{}".format(cf.node_labels[preds[i]], round(float(values[i]), 2))
            color = (0, 0, 255)

            info = boxes[i]
            box = np.array(
                [
                    [int(info[0]), int(info[1])],
                    [int(info[2]), int(info[3])],
                    [int(info[4]), int(info[5])],
                    [int(info[6]), int(info[7])],
                ]
            )
            cv2.polylines(vis_img, [box], 1, (255, 0, 0))
            cv2.putText(
                vis_img,
                msg,
                (int(info[0]), int(info[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    imageio.imwrite(save_path, vis_img)
    return vis_img
