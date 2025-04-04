import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence
import dgl
import dgl.function as fn
from transformers import LayoutLMv3Model, LayoutLMv3Tokenizer

import numpy as np


from models.kie.graph_norm import GraphNorm


"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""


class MLPReadout(nn.Module):
    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [
            nn.Linear(input_dim // 2**l, input_dim // 2 ** (l + 1), bias=True)
            for l in range(L)
        ]
        list_FC_layers.append(nn.Linear(input_dim // 2**L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y


class GatedGCNLayer(nn.Module):
    """
    Gated GCN layer for node and edge feature update
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.A = nn.Linear(input_dim, output_dim)
        self.B = nn.Linear(input_dim, output_dim)
        self.C = nn.Linear(input_dim, output_dim)
        self.D = nn.Linear(input_dim, output_dim)
        self.E = nn.Linear(input_dim, output_dim)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)

    def message_func(self, edges):
        Bh_j = edges.src["Bh"]
        e_ij = edges.data["Ce"] + edges.src["Dh"] + edges.dst["Eh"]
        edges.data["e"] = e_ij
        return {"Bh_j": Bh_j, "e_ij": e_ij}

    def reduce_func(self, nodes):
        Ah_i = nodes.data["Ah"]
        Bh_j = nodes.mailbox["Bh_j"]
        e = nodes.mailbox["e_ij"]
        sigma_ij = torch.sigmoid(e)
        h = Ah_i + torch.sum(sigma_ij * Bh_j, dim=1) / (
            torch.sum(sigma_ij, dim=1) + 1e-6
        )
        return {"h": h}

    def forward(self, g, h, e):
        # Lưu device của input tensors
        h_device = h.device
        e_device = e.device

        # Chuyển features về CPU để xử lý với DGL
        h_cpu = h.to("cpu")
        e_cpu = e.to("cpu")

        # Đảm bảo graph ở trên CPU
        g = g.to("cpu")

        # Tính toán trên CPU
        g.ndata["h"] = h_cpu
        g.ndata["Ah"] = self.A.weight.data.to("cpu") @ h_cpu.t()
        if self.A.bias is not None:
            g.ndata["Ah"] = g.ndata["Ah"].t() + self.A.bias.data.to("cpu")
        else:
            g.ndata["Ah"] = g.ndata["Ah"].t()

        g.ndata["Bh"] = self.B.weight.data.to("cpu") @ h_cpu.t()
        if self.B.bias is not None:
            g.ndata["Bh"] = g.ndata["Bh"].t() + self.B.bias.data.to("cpu")
        else:
            g.ndata["Bh"] = g.ndata["Bh"].t()

        g.ndata["Dh"] = self.D.weight.data.to("cpu") @ h_cpu.t()
        if self.D.bias is not None:
            g.ndata["Dh"] = g.ndata["Dh"].t() + self.D.bias.data.to("cpu")
        else:
            g.ndata["Dh"] = g.ndata["Dh"].t()

        g.ndata["Eh"] = self.E.weight.data.to("cpu") @ h_cpu.t()
        if self.E.bias is not None:
            g.ndata["Eh"] = g.ndata["Eh"].t() + self.E.bias.data.to("cpu")
        else:
            g.ndata["Eh"] = g.ndata["Eh"].t()

        g.edata["e"] = e_cpu
        g.edata["Ce"] = self.C.weight.data.to("cpu") @ e_cpu.t()
        if self.C.bias is not None:
            g.edata["Ce"] = g.edata["Ce"].t() + self.C.bias.data.to("cpu")
        else:
            g.edata["Ce"] = g.edata["Ce"].t()

        g.update_all(self.message_func, self.reduce_func)

        h = g.ndata["h"]
        e = g.edata["e"]

        # Batch norm trên CPU
        h = self.bn_node_h.weight.data.to("cpu") * (
            h - self.bn_node_h.running_mean.to("cpu")
        ) / torch.sqrt(
            self.bn_node_h.running_var.to("cpu") + self.bn_node_h.eps
        ) + self.bn_node_h.bias.data.to(
            "cpu"
        )
        e = self.bn_node_e.weight.data.to("cpu") * (
            e - self.bn_node_e.running_mean.to("cpu")
        ) / torch.sqrt(
            self.bn_node_e.running_var.to("cpu") + self.bn_node_e.eps
        ) + self.bn_node_e.bias.data.to(
            "cpu"
        )

        h = F.relu(h)
        e = F.relu(e)

        # Chuyển kết quả về device ban đầu
        h = h.to(h_device)
        e = e.to(e_device)

        return h, e

    def __repr__(self):
        return "{}(in_channels={}, out_channels={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )


class DenseLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # self.bn = nn.BatchNorm1d(in_dim)
        self.bn = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, feat):
        feat = self.bn(feat)
        feat = F.relu(feat)
        feat = self.linear(feat)
        return feat


class GatedGCNNet(nn.Module):
    """
    GatedGCN network with LayoutXLM integration for KIE task
    """

    def __init__(self, net_params):
        super().__init__()
        # Convert string device to torch.device object
        self.device = torch.device(net_params.get("device", "cpu"))
        self.use_cuda = self.device.type == "cuda"

        # DGL luôn chạy trên CPU
        self.dgl_device = torch.device("cpu")

        in_dim_text = net_params["in_dim_text"]
        in_dim_node = net_params["in_dim_node"]
        in_dim_edge = net_params["in_dim_edge"]
        hidden_dim = net_params["hidden_dim"]
        out_dim = net_params["out_dim"]
        n_classes = net_params["n_classes"]
        dropout = net_params["dropout"]
        n_layers = net_params["L"]
        self.readout = net_params["readout"]
        self.batch_norm = net_params["batch_norm"]
        self.residual = net_params["residual"]
        self.in_feat_dropout = net_params["in_feat_dropout"]

        # LayoutXLM for text feature extraction
        self.layoutxlm = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base")
        self.tokenizer = LayoutLMv3Tokenizer.from_pretrained(
            "microsoft/layoutlmv3-base"
        )

        # Move model to device
        self.layoutxlm = self.layoutxlm.to(self.device)
        self.tokenizer.model_max_length = 512

        # Node and edge encoders
        self.node_encoder = nn.Linear(in_dim_node + in_dim_text, hidden_dim)
        self.edge_encoder = nn.Linear(in_dim_edge, hidden_dim)

        # GatedGCN layers
        self.layers = nn.ModuleList(
            [GatedGCNLayer(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )

        # Output layers
        self.MLP_layer = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, n_classes),
        )

        # Move all layers to device
        self.to(self.device)

    def to(self, device):
        # Convert string device to torch.device object if needed
        if isinstance(device, str):
            device = torch.device(device)
        super().to(device)
        self.device = device
        self.use_cuda = device.type == "cuda"

        # Move LayoutXLM to device
        self.layoutxlm = self.layoutxlm.to(device)

        # Move all other layers to device
        self.node_encoder = self.node_encoder.to(device)
        self.edge_encoder = self.edge_encoder.to(device)
        for layer in self.layers:
            layer.to(device)
        self.MLP_layer = self.MLP_layer.to(device)

        return self

    def forward(self, boxes, texts):
        batch_size = len(boxes)
        batch_graphs = []
        node_features = []
        edge_features = []

        # Process each sample in the batch
        for i in range(batch_size):
            n_nodes = boxes[i].size(0)

            # Create graph on CPU (DGL không hỗ trợ CUDA)
            g = dgl.DGLGraph()
            g.add_nodes(n_nodes)

            # Add self-loops
            g.add_edges(g.nodes(), g.nodes())

            # Add edges between all nodes
            src, dst = [], []
            for i1 in range(n_nodes):
                for i2 in range(n_nodes):
                    if i1 != i2:
                        src.append(i1)
                        dst.append(i2)
            g.add_edges(src, dst)

            # Compute edge features (relative spatial information)
            e_feats = []
            for s, d in zip(src + list(range(n_nodes)), dst + list(range(n_nodes))):
                box1 = boxes[i][s].to(self.device)
                box2 = boxes[i][d].to(self.device)

                # Calculate center points
                if box1.dim() == 0:
                    box1 = box1.view(1)
                if box2.dim() == 0:
                    box2 = box2.view(1)

                try:
                    center1 = torch.tensor(
                        [
                            (
                                box1[0].item()
                                + box1[2].item()
                                + box1[4].item()
                                + box1[6].item()
                            )
                            / 4,
                            (
                                box1[1].item()
                                + box1[3].item()
                                + box1[5].item()
                                + box1[7].item()
                            )
                            / 4,
                        ],
                        device=self.device,
                    )
                    center2 = torch.tensor(
                        [
                            (
                                box2[0].item()
                                + box2[2].item()
                                + box2[4].item()
                                + box2[6].item()
                            )
                            / 4,
                            (
                                box2[1].item()
                                + box2[3].item()
                                + box2[5].item()
                                + box2[7].item()
                            )
                            / 4,
                        ],
                        device=self.device,
                    )
                except (IndexError, AttributeError):
                    center1 = torch.tensor([0.0, 0.0], device=self.device)
                    center2 = torch.tensor([0.0, 0.0], device=self.device)

                # Compute relative position
                rel_pos = center2 - center1
                e_feats.append(rel_pos)

            e_feats = torch.stack(e_feats)

            # Get text features from LayoutXLM
            text_list = []
            bbox_list = []
            for j in range(n_nodes):
                text_tensor = texts[i][j]
                if isinstance(text_tensor, torch.Tensor):
                    if text_tensor.dim() == 0:
                        text = str(text_tensor.item())
                    else:
                        text = " ".join([str(x.item()) for x in text_tensor])
                else:
                    text = str(text_tensor)
                text_list.append(text)

                box = boxes[i][j].to(self.device)
                if box.dim() == 0:
                    box = box.view(1)

                try:
                    bbox = [
                        int(
                            min(
                                box[0].item(),
                                box[2].item(),
                                box[4].item(),
                                box[6].item(),
                            )
                        ),
                        int(
                            min(
                                box[1].item(),
                                box[3].item(),
                                box[5].item(),
                                box[7].item(),
                            )
                        ),
                        int(
                            max(
                                box[0].item(),
                                box[2].item(),
                                box[4].item(),
                                box[6].item(),
                            )
                        ),
                        int(
                            max(
                                box[1].item(),
                                box[3].item(),
                                box[5].item(),
                                box[7].item(),
                            )
                        ),
                    ]
                except (IndexError, AttributeError):
                    bbox = [0, 0, 100, 100]

                bbox_list.append(bbox)

            # Tokenize text with bounding boxes
            text_inputs = self.tokenizer(
                text_list,
                boxes=bbox_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            for key in text_inputs:
                text_inputs[key] = text_inputs[key].to(self.device)

            text_inputs["bbox"] = text_inputs["bbox"].long()

            text_outputs = self.layoutxlm(**text_inputs)
            text_feats = text_outputs.last_hidden_state[:, 0, :]

            box_feats = boxes[i].to(self.device)
            if box_feats.dim() == 1:
                box_feats = box_feats.view(1, -1)

            if text_feats.dim() == 1:
                text_feats = text_feats.view(1, -1)

            text_feats = text_feats.expand(box_feats.size(0), -1)

            expected_box_features = 8
            if box_feats.size(1) != expected_box_features:
                if box_feats.size(1) < expected_box_features:
                    padding = torch.zeros(
                        box_feats.size(0),
                        expected_box_features - box_feats.size(1),
                        device=self.device,
                    )
                    box_feats = torch.cat([box_feats, padding], dim=1)
                else:
                    box_feats = box_feats[:, :expected_box_features]

            expected_text_features = 768
            if text_feats.size(1) != expected_text_features:
                if text_feats.size(1) < expected_text_features:
                    padding = torch.zeros(
                        text_feats.size(0),
                        expected_text_features - text_feats.size(1),
                        device=self.device,
                    )
                    text_feats = torch.cat([text_feats, padding], dim=1)
                else:
                    text_feats = text_feats[:, :expected_text_features]

            h_feats = torch.cat([box_feats, text_feats], dim=1)

            batch_graphs.append(g)
            node_features.append(h_feats)
            edge_features.append(e_feats)

        # DGL luôn chạy trên CPU
        batch_graph = dgl.batch(batch_graphs)
        # Không cần chuyển batch_graph đến device vì DGL không hỗ trợ CUDA

        h = torch.cat(node_features, dim=0).to(self.device)
        e = torch.cat(edge_features, dim=0).to(self.device)

        if h.size(0) != batch_graph.num_nodes():
            h = h.view(batch_graph.num_nodes(), -1)

        expected_features = self.node_encoder.weight.size(1)
        if h.size(1) != expected_features:
            if h.size(1) < expected_features:
                padding = torch.zeros(
                    h.size(0), expected_features - h.size(1), device=self.device
                )
                h = torch.cat([h, padding], dim=1)
            else:
                h = h[:, :expected_features]

        # Chuyển features về CPU để xử lý với DGL
        h_cpu = h.to("cpu")
        e_cpu = e.to("cpu")

        # Tính toán node và edge features trên CPU
        h_cpu = self.node_encoder.weight.data.to("cpu") @ h_cpu.t()
        if self.node_encoder.bias is not None:
            h_cpu = h_cpu.t() + self.node_encoder.bias.data.to("cpu")
        else:
            h_cpu = h_cpu.t()

        e_cpu = self.edge_encoder.weight.data.to("cpu") @ e_cpu.t()
        if self.edge_encoder.bias is not None:
            e_cpu = e_cpu.t() + self.edge_encoder.bias.data.to("cpu")
        else:
            e_cpu = e_cpu.t()

        # Apply dropout
        h_cpu = F.dropout(h_cpu, self.in_feat_dropout, training=self.training)

        # GatedGCN layers
        for conv in self.layers:
            h_cpu, e_cpu = conv(batch_graph, h_cpu, e_cpu)

        # Output trên CPU
        h_out_cpu = h_cpu
        for layer in self.MLP_layer:
            if isinstance(layer, nn.Linear):
                h_out_cpu = layer.weight.data.to("cpu") @ h_out_cpu.t()
                if layer.bias is not None:
                    h_out_cpu = h_out_cpu.t() + layer.bias.data.to("cpu")
                else:
                    h_out_cpu = h_out_cpu.t()
            elif isinstance(layer, nn.Dropout):
                h_out_cpu = F.dropout(h_out_cpu, layer.p, training=self.training)
            elif isinstance(layer, nn.ReLU):
                h_out_cpu = F.relu(h_out_cpu)

        # Chuyển kết quả về device ban đầu
        h_out = h_out_cpu.to(self.device)

        return h_out

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss


if __name__ == "__main__":

    net_params = {}
    net_params["in_dim"] = 1
    net_params["hidden_dim"] = 256
    net_params["out_dim"] = 256
    net_params["n_classes"] = 5
    net_params["in_feat_dropout"] = 0.1
    net_params["dropout"] = 0.1
    net_params["L"] = 5
    net_params["readout"] = True
    net_params["graph_norm"] = True
    net_params["batch_norm"] = True
    net_params["residual"] = True
    net_params["device"] = "cuda"

    net = GatedGCNNet(net_params)
    print(net)
