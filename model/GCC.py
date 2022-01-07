# -*- coding: utf-8 -*-
# @Time       : 2022/01/04 16:43:57
# @Author     : Zhan Genze <947783684@qq.com>
# @Project    : gzzhan
# @Description:  idea from http://keg.cs.tsinghua.edu.cn/jietang/publications/KDD20-Qiu-et-al-GCC-GNN-pretrain.pdf
#                pretrain med and disease from random walk of the same node

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgllife.model import gnn

from model.gnn.gat import GATEncoder
from model.gnn.gcn import GCNEncoder
from model.gnn.gin import GINEncoder


class GCC(nn.Module):
    def __init__(self,*args):
        super(GCC, self).__init__()
        gnn_model = args[0]['gnn']['encoder']
        if gnn_model == "gat":
            self.model = GATEncoder(**args[0]['gnn']['config'])
        elif gnn_model == "gin":
            self.model = GINEncoder(**args[0]['gnn']['config'])
        elif gnn_model == "gcn":
            self.model = GCNEncoder(**args[0]['gnn']['config'])

    def forward(self, g, node_feats):
        return self.model(g, node_feats)
