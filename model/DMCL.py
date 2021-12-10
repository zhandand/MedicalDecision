# -*- coding: utf-8 -*-
# @Time       : 2021/12/08 10:46:55
# @Author     : Zhan Genze <947783684@qq.com>
# @Project    : gzzhan
# @Description:  针对药物和疾病的contrastive learning


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DMCL(nn.Module):
    def __init__(self,*args):
        super(DMCL,self).__init__()
        vocab = args[1]['vocab']
        diag_voc = vocab['diag_voc']
        med_voc = vocab['med_voc']
        self.dim = args[0]['dim']
        self.diag_embedding = nn.Embedding(len(diag_voc.idx2word),self.dim)
        self.med_embedding = nn.Embedding(len(med_voc.idx2word),self.dim)
        # projection head
        self.g = nn.Sequential(nn.Linear(self.dim, 256, bias=False),
                            #    nn.BatchNorm1d(256),
                               nn.ReLU(inplace=True),
                               nn.Linear(256, self.dim, bias=True))


    def forward(self,x):
        # 药物索引
        single_diag = self.diag_embedding(x)
        sim = torch.mm(self.g(single_diag) ,self.g(self.med_embedding.weight).T)
        return sim

