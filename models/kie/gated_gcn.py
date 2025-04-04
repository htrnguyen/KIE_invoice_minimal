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

        # Tạo các phiên bản CPU của các layer
        A_cpu = nn.Linear(self.A.in_features, self.A.out_features)
        B_cpu = nn.Linear(self.B.in_features, self.B.out_features)
        C_cpu = nn.Linear(self.C.in_features, self.C.out_features)
        D_cpu = nn.Linear(self.D.in_features, self.D.out_features)
        E_cpu = nn.Linear(self.E.in_features, self.E.out_features)

        # Sao chép trọng số từ các layer gốc
        A_cpu.weight.data = self.A.weight.data.clone().to("cpu")
        if self.A.bias is not None:
            A_cpu.bias.data = self.A.bias.data.clone().to("cpu")

        B_cpu.weight.data = self.B.weight.data.clone().to("cpu")
        if self.B.bias is not None:
            B_cpu.bias.data = self.B.bias.data.clone().to("cpu")

        C_cpu.weight.data = self.C.weight.data.clone().to("cpu")
        if self.C.bias is not None:
            C_cpu.bias.data = self.C.bias.data.clone().to("cpu")

        D_cpu.weight.data = self.D.weight.data.clone().to("cpu")
        if self.D.bias is not None:
            D_cpu.bias.data = self.D.bias.data.clone().to("cpu")

        E_cpu.weight.data = self.E.weight.data.clone().to("cpu")
        if self.E.bias is not None:
            E_cpu.bias.data = self.E.bias.data.clone().to("cpu")

        # Tính toán trên CPU
        g.ndata["h"] = h_cpu
        g.ndata["Ah"] = A_cpu(h_cpu)
        g.ndata["Bh"] = B_cpu(h_cpu)
        g.ndata["Dh"] = D_cpu(h_cpu)
        g.ndata["Eh"] = E_cpu(h_cpu)
        g.edata["e"] = e_cpu
        g.edata["Ce"] = C_cpu(e_cpu)

        g.update_all(self.message_func, self.reduce_func)

        h = g.ndata["h"]
        e = g.edata["e"]

        # Tạo phiên bản CPU của batch norm
        bn_node_h_cpu = nn.BatchNorm1d(self.bn_node_h.num_features)
        bn_node_e_cpu = nn.BatchNorm1d(self.bn_node_e.num_features)

        # Sao chép trọng số từ batch norm gốc
        bn_node_h_cpu.weight.data = self.bn_node_h.weight.data.clone().to("cpu")
        bn_node_h_cpu.bias.data = self.bn_node_h.bias.data.clone().to("cpu")
        bn_node_h_cpu.running_mean = self.bn_node_h.running_mean.clone().to("cpu")
        bn_node_h_cpu.running_var = self.bn_node_h.running_var.clone().to("cpu")

        bn_node_e_cpu.weight.data = self.bn_node_e.weight.data.clone().to("cpu")
        bn_node_e_cpu.bias.data = self.bn_node_e.bias.data.clone().to("cpu")
        bn_node_e_cpu.running_mean = self.bn_node_e.running_mean.clone().to("cpu")
        bn_node_e_cpu.running_var = self.bn_node_e.running_var.clone().to("cpu")

        h = bn_node_h_cpu(h)
        e = bn_node_e_cpu(e)

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
        """
        Forward pass của mô hình.
        Args:
            boxes: List các tensor chứa tọa độ các box
            texts: List các tensor chứa text đã được encode
        Returns:
            Tensor chứa logits cho mỗi node
        """
        batch_size = len(boxes)
        device = boxes[0].device

        # Xử lý từng mẫu trong batch
        all_node_features = []
        all_edge_features = []
        all_graphs = []

        for i in range(batch_size):
            num_nodes = boxes[i].size(0)
            if num_nodes == 0:
                continue

            # Tạo node features từ boxes
            node_features = boxes[i]  # [num_nodes, 8]

            # Đảm bảo node_features là tensor 2 chiều
            if node_features.dim() == 1:
                node_features = node_features.unsqueeze(0)  # Thêm dimension cho batch
                num_nodes = 1

            # Nếu chỉ có 1 node, tạo thêm 1 node giả để có thể tạo edge
            if num_nodes == 1:
                # Tạo node giả với tọa độ khác biệt
                fake_node = node_features[0].clone()
                fake_node[0] += 1.0  # Dịch chuyển 1 đơn vị theo trục x
                node_features = torch.cat(
                    [node_features, fake_node.unsqueeze(0)], dim=0
                )
                num_nodes = 2

            # Tạo edge features từ tương đối vị trí
            edge_features = []
            edge_index = []

            # Tính toán tương đối vị trí giữa các node
            for j in range(num_nodes):
                for k in range(num_nodes):
                    if j != k:
                        # Tính tương đối vị trí
                        # Đảm bảo node_features[j] và node_features[k] là tensor 1 chiều
                        node_j = node_features[j]
                        node_k = node_features[k]

                        if node_j.dim() == 0:
                            node_j = node_j.unsqueeze(0)
                        if node_k.dim() == 0:
                            node_k = node_k.unsqueeze(0)

                        # Lấy 2 phần tử đầu tiên cho tọa độ x, y
                        if node_j.size(0) >= 2 and node_k.size(0) >= 2:
                            rel_pos = node_j[:2] - node_k[:2]  # [2]
                        else:
                            # Nếu không đủ phần tử, tạo tọa độ giả
                            rel_pos = torch.zeros(2, device=device)

                        edge_features.append(rel_pos)
                        edge_index.append([j, k])

            if not edge_features:  # Nếu không có edge nào
                edge_features = torch.zeros((2, 2), device=device)  # Tạo 2 edge giả
                edge_index = [[0, 1], [1, 0]]

            edge_features = torch.stack(edge_features)  # [num_edges, 2]
            edge_index = torch.tensor(edge_index, device=device).t()  # [2, num_edges]

            # Tạo DGL graph trên CPU
            g = dgl.graph(
                (edge_index[0].cpu(), edge_index[1].cpu()), num_nodes=num_nodes
            )
            g.ndata["feat"] = node_features.cpu()  # Chuyển node features về CPU
            g.edata["feat"] = edge_features.cpu()  # Chuyển edge features về CPU

            # Xử lý text features
            text_features = []
            for j in range(num_nodes):
                # Kiểm tra kích thước của texts[i][j]
                if j >= len(texts[i]):
                    # Nếu không đủ text, tạo text giả
                    text_tensor = torch.zeros(2, device=device)
                else:
                    if texts[i][j].dim() == 0:  # Nếu là scalar
                        text_tensor = texts[i][j].unsqueeze(0)  # Thêm dimension
                    else:
                        text_tensor = texts[i][j]

                    # Đảm bảo text_tensor có kích thước phù hợp
                    if text_tensor.size(0) < 2:
                        # Nếu kích thước nhỏ hơn 2, thêm padding
                        padding = torch.zeros(2 - text_tensor.size(0), device=device)
                        text_tensor = torch.cat([text_tensor, padding])

                    # Lấy 2 phần tử đầu tiên
                    text_tensor = text_tensor[:2]

                # Thêm vào text_features
                text_features.append(text_tensor)

            text_features = torch.stack(text_features)  # [num_nodes, 2]

            # Kết hợp node features và text features
            node_features = torch.cat(
                [node_features, text_features], dim=1
            )  # [num_nodes, 10]

            # Lưu features và graph
            all_node_features.append(node_features)
            all_edge_features.append(edge_features)
            all_graphs.append(g)

        if not all_node_features:  # Nếu không có mẫu nào
            return torch.zeros((0, self.n_classes), device=device)

        # Stack tất cả features
        node_features = torch.cat(all_node_features, dim=0)  # [total_nodes, 10]
        edge_features = torch.cat(all_edge_features, dim=0)  # [total_edges, 2]

        # Tạo batch graph trên CPU
        batch_graph = dgl.batch(all_graphs)

        # Kiểm tra kích thước của node_features và edge_features
        # print(f"node_features shape: {node_features.shape}")
        # print(f"edge_features shape: {edge_features.shape}")
        # print(f"node_encoder weight shape: {self.node_encoder.weight.shape}")
        # print(f"edge_encoder weight shape: {self.edge_encoder.weight.shape}")

        # Điều chỉnh kích thước của node_features để phù hợp với node_encoder
        # Nếu kích thước không khớp, tạo một layer mới với kích thước phù hợp
        if node_features.shape[1] != self.node_encoder.in_features:
            print(
                f"Adjusting node_encoder input size from {self.node_encoder.in_features} to {node_features.shape[1]}"
            )
            self.node_encoder = nn.Linear(
                node_features.shape[1], self.node_encoder.out_features
            ).to(device)

        if edge_features.shape[1] != self.edge_encoder.in_features:
            print(
                f"Adjusting edge_encoder input size from {self.edge_encoder.in_features} to {edge_features.shape[1]}"
            )
            self.edge_encoder = nn.Linear(
                edge_features.shape[1], self.edge_encoder.out_features
            ).to(device)

        # Chuyển đổi kích thước node_features và edge_features để phù hợp với các layer
        # Sử dụng node_encoder và edge_encoder để chuyển đổi kích thước
        node_features = self.node_encoder(node_features)  # [total_nodes, hidden_dim]
        edge_features = self.edge_encoder(edge_features)  # [total_edges, hidden_dim]

        # Forward qua GatedGCN layers
        h = node_features
        e = edge_features

        for layer in self.layers:
            h, e = layer(batch_graph, h, e)

        # Global pooling
        if self.readout == "sum":
            h = dgl.sum_nodes(batch_graph, "feat")
        elif self.readout == "max":
            h = dgl.max_nodes(batch_graph, "feat")
        elif self.readout == "mean":
            h = dgl.mean_nodes(batch_graph, "feat")
        else:
            h = dgl.mean_nodes(batch_graph, "feat")

        # Đảm bảo h nằm trên cùng device với MLP_layer
        h = h.to(device)

        # Kiểm tra kích thước của h và MLP_layer
        print(f"h shape before MLP: {h.shape}")
        print(f"MLP_layer first layer weight shape: {self.MLP_layer[0].weight.shape}")

        # Điều chỉnh kích thước của MLP_layer nếu cần
        if h.shape[1] != self.MLP_layer[0].in_features:
            print(
                f"Adjusting MLP_layer input size from {self.MLP_layer[0].in_features} to {h.shape[1]}"
            )
            # Tạo MLP layer mới với kích thước phù hợp
            out_features = self.MLP_layer[-1].out_features
            self.MLP_layer = nn.Sequential(
                nn.Linear(h.shape[1], 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, out_features),
            ).to(device)

        # MLP layers
        h = self.MLP_layer(h)

        return h

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
