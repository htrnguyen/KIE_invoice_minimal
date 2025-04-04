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
    # Use the device passed to the function
    dgl_device = device

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
        # Get polygon coordinates
        poly = cell.get("poly", [])
        if isinstance(poly, np.ndarray):
            poly = poly.tolist()

        # Normalize coordinates
        x_coords = [(x - x_min) / (x_max - x_min) for x in poly[0::2]]
        y_coords = [(y - y_min) / (y_max - y_min) for y in poly[1::2]]

        # Create box features
        box_feats = []
        for i in range(len(x_coords)):
            box_feats.extend([x_coords[i], y_coords[i]])

        # Add width and height
        width = (max(x_coords) - min(x_coords)) if x_coords else 0
        height = (max(y_coords) - min(y_coords)) if y_coords else 0
        box_feats.extend([width, height])

        # Convert to tensor and add to batch
        box_tensor = torch.tensor(box_feats, dtype=torch.float32).to(device)
        batch_boxes.append(box_tensor)

        # Process text
        text = cell.get("vietocr_text", "")
        text_encode = make_text_encode(text)
        text_tensor = torch.tensor(text_encode, dtype=torch.long).to(device)
        batch_texts.append(text_tensor)

    # Nếu không có cell nào, trả về kết quả rỗng
    if not batch_boxes:
        return np.array([]), []

    # Nếu chỉ có 1 cell, thêm một cell giả để tránh lỗi
    if len(batch_boxes) == 1:
        # Tạo box giả với tọa độ khác biệt
        fake_box = batch_boxes[0].clone()
        fake_box[0] += 1.0  # Dịch chuyển 1 đơn vị theo trục x
        batch_boxes.append(fake_box)

        # Tạo text giả
        fake_text = batch_texts[0].clone()
        batch_texts.append(fake_text)

    # Run inference
    with torch.no_grad():
        scores = gcn_net(batch_boxes, batch_texts)

    # Convert scores to numpy for post-processing
    scores = scores.cpu().numpy()

    return scores, batch_boxes


@timer
def postprocess_scores(batch_scores, score_ths=0.98, get_max=False):
    values, preds = [], []

    # Check if batch_scores is a PyTorch tensor or NumPy array
    if isinstance(batch_scores, torch.Tensor):
        batch_scores = batch_scores.cpu().softmax(1)
        for score in batch_scores:
            _score = score.detach().numpy()
            values.append(_score.max())
            pred_index = np.argmax(_score)
            if get_max:
                preds.append(pred_index)
            else:
                if pred_index != 0 and _score.max() >= score_ths:
                    preds.append(pred_index)
                else:
                    preds.append(0)
    else:
        # If it's already a NumPy array, apply softmax manually
        # Compute softmax along axis 1
        exp_scores = np.exp(batch_scores - np.max(batch_scores, axis=1, keepdims=True))
        softmax_scores = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        for score in softmax_scores:
            values.append(score.max())
            pred_index = np.argmax(score)
            if get_max:
                preds.append(pred_index)
            else:
                if pred_index != 0 and score.max() >= score_ths:
                    preds.append(pred_index)
                else:
                    preds.append(0)

    return values, preds


@timer
def postprocess_write_info(merged_cells, preds, text_key="vietocr_text"):
    kie_info = dict()
    preds = np.array(preds)

    # Process each label type
    for i in range(1, len(cf.node_labels)):  # Skip 'OTHER' class
        indexes = np.where(preds == i)[0]
        if len(indexes) > 0:
            # Filter out invalid indices
            valid_indexes = [idx for idx in indexes if idx < len(merged_cells)]

            if len(valid_indexes) > 0:
                # For label fields (MFG_LABEL, EXP_LABEL, WEIGHT_LABEL), we want to keep them separate
                if cf.node_labels[i].endswith("_LABEL"):
                    text_output = " ".join(
                        merged_cells[index][text_key] for index in valid_indexes
                    )
                    kie_info[cf.node_labels[i]] = text_output
                else:
                    # For value fields (MFG, EXP, WEIGHT), we want to concatenate them
                    text_output = " ".join(
                        merged_cells[index][text_key] for index in valid_indexes
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
