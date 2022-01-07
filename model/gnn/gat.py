# -*- coding: utf-8 -*-
# @Time       : 2022/01/04 20:38:05
# @Author     : Zhan Genze <947783684@qq.com>
# @Project    : gzzhan
# @Description:  gat encoder

import torch.nn as nn

from dgllife.model import GAT
from dgllife.model.readout import WeightedSumAndMax


class GATEncoder(nn.Module):
    def __init__(self,  in_feats, hidden_feats=None, num_heads=None, feat_drops=None,
                 attn_drops=None, alphas=None, residuals=None, agg_modes=None, activations=None,
                 biases=None, ):
        super(GATEncoder).__init__()
        self.gnn = GAT(in_feats=in_feats,
                       hidden_feats=hidden_feats,
                       num_heads=num_heads,
                       feat_drops=feat_drops,
                       attn_drops=attn_drops,
                       alphas=alphas,
                       residuals=residuals,
                       agg_modes=agg_modes,
                       activations=activations,
                       biases=biases)

        if self.gnn.agg_modes[-1] == 'flatten':
            gnn_out_feats = self.gnn.hidden_feats[-1] * self.gnn.num_heads[-1]
        else:
            gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)

    def forward(self, bg, feats):
        node_feats = self.gnn(bg, feats)
        graph_feats = self.readout(bg, node_feats)
        return graph_feats