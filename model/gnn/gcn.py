# -*- coding: utf-8 -*-
# @Time       : 2022/01/04 20:08:19
# @Author     : Zhan Genze <947783684@qq.com>
# @Project    : gzzhan
# @Description: GCN 编码整张图
import torch.nn as nn

from dgllife.model import GCN
from dgllife.model.readout import WeightedSumAndMax


class GCNEncoder(nn.Module):
    def __init__(self, in_feats, hidden_feats=None, gnn_norm=None, activation=None,
                 residual=None, batchnorm=None, dropout=None):
        super(GCNEncoder).__init__()
        self.gnn = GCN(in_feats=in_feats,
                       hidden_feats=hidden_feats,
                       gnn_norm=gnn_norm,
                       activation=activation,
                       residual=residual,
                       batchnorm=batchnorm,
                       dropout=dropout)
        gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)

    def forward(self, bg, feats):
        node_feats = self.gnn(bg, feats)
        graph_feats = self.readout(bg, node_feats)
        return graph_feats
