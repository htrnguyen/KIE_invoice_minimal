import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence
import dgl
import dgl.function as fn
from transformers import LayoutLMv3Model

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

    def __init__(
        self, input_dim, output_dim, dropout=0.0, batch_norm=True, residual=True
    ):
        super().__init__()

        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False

        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)

        if self.batch_norm:
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
        sigma = torch.sigmoid(e)
        h = Ah_i + torch.sum(sigma * Bh_j, dim=1) / (torch.sum(sigma, dim=1) + 1e-6)
        return {"h": h}

    def forward(self, g, h, e):
        h_in = h
        e_in = e

        g.ndata["h"] = h
        g.ndata["Ah"] = self.A(h)
        g.ndata["Bh"] = self.B(h)
        g.ndata["Dh"] = self.D(h)
        g.ndata["Eh"] = self.E(h)
        g.edata["e"] = e
        g.edata["Ce"] = self.C(e)

        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata["h"]
        e = g.edata["e"]

        if self.batch_norm:
            h = self.bn_node_h(h)
            e = self.bn_node_e(e)

        h = F.relu(h)
        e = F.relu(e)

        if self.residual:
            h = h_in + h
            e = e_in + e

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

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
        self.device = net_params["device"]
        self.in_feat_dropout = net_params["in_feat_dropout"]

        # LayoutXLM for text feature extraction
        self.layoutxlm = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base")
        self.text_projection = nn.Linear(
            768, hidden_dim
        )  # 768 is LayoutXLM hidden size

        # Node and edge encoders
        self.node_encoder = nn.Linear(in_dim_node + hidden_dim, hidden_dim)
        self.edge_encoder = nn.Linear(in_dim_edge, hidden_dim)

        # GatedGCN layers
        self.layers = nn.ModuleList(
            [
                GatedGCNLayer(
                    hidden_dim, hidden_dim, dropout, self.batch_norm, self.residual
                )
                for _ in range(n_layers)
            ]
        )

        # Output layers
        self.MLP_layer = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, n_classes),
        )

    def forward(self, boxes, texts):
        batch_size = len(boxes)
        batch_graphs = []
        node_features = []
        edge_features = []

        # Process each sample in the batch
        for i in range(batch_size):
            n_nodes = boxes[i].size(0)

            # Create graph
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
                box1 = boxes[i][s]
                box2 = boxes[i][d]

                # Calculate center points
                center1 = torch.tensor(
                    [
                        (box1[0] + box1[2] + box1[4] + box1[6]) / 4,
                        (box1[1] + box1[3] + box1[5] + box1[7]) / 4,
                    ]
                ).to(self.device)
                center2 = torch.tensor(
                    [
                        (box2[0] + box2[2] + box2[4] + box2[6]) / 4,
                        (box2[1] + box2[3] + box2[5] + box2[7]) / 4,
                    ]
                ).to(self.device)

                # Compute relative position
                rel_pos = center2 - center1
                e_feats.append(rel_pos)

            e_feats = torch.stack(e_feats)

            # Get text features from LayoutXLM
            text_inputs = self.layoutxlm.tokenizer(
                texts[i], padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

            text_outputs = self.layoutxlm(**text_inputs)
            text_feats = text_outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
            text_feats = self.text_projection(text_feats)

            # Node features: concatenate box coordinates and text features
            box_feats = boxes[i]
            h_feats = torch.cat([box_feats, text_feats], dim=1)

            batch_graphs.append(g)
            node_features.append(h_feats)
            edge_features.append(e_feats)

        # Batch graphs
        batch_graph = dgl.batch(batch_graphs).to(self.device)

        # Concatenate features
        h = torch.cat(node_features, dim=0)
        e = torch.cat(edge_features, dim=0)

        # Initial node and edge encoders
        h = self.node_encoder(h)
        e = self.edge_encoder(e)

        # Apply dropout
        h = F.dropout(h, self.in_feat_dropout, training=self.training)

        # GatedGCN layers
        for conv in self.layers:
            h, e = conv(batch_graph, h, e)

        # Output
        h_out = self.MLP_layer(h)

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
