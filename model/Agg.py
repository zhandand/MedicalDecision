# -*- coding: utf-8 -*-
# @Time       : 2021/12/03 15:11:48
# @Author     : Zhan Genze <947783684@qq.com>
# @Project    : gzzhan
# @Description:  对于药物和疾病构建embedding，
#                病人的表示为 历史的药物和疾病平均 +当前疾病
#                加mlp实现多分类

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.DMCL import DMCL

class Agg(nn.Module):
    def __init__(self,*args):
        super(Agg,self).__init__()
        diag_voc = args[1]['vocab']['diag_voc']
        med_voc = args[1]['vocab']['med_voc']
        self.dim = args[0]['dim']
        # TODO: 加入预训练embedding
        if 'pretrain' in args[0].keys():
            pretrain_model = args[0]['pretrain'].model
            # 默认为DMCL
            self.diag_embedding = pretrain_model.diag_embedding
            self.med_embedding = pretrain_model.med_embedding
        else:
            self.diag_embedding = nn.Embedding(len(diag_voc.idx2word),self.dim)
            self.med_embedding = nn.Embedding(len(med_voc.idx2word),self.dim)

        self.dense1 = nn.Sequential(nn.Linear(self.dim, self.dim),
                                   nn.ReLU())
        self.dense2 = nn.Sequential(nn.Linear(self.dim, self.dim),
                                   nn.ReLU())
        self.cls = nn.Sequential(nn.Linear(3*self.dim, 2*self.dim),
                                 nn.ReLU(), nn.Linear(2*self.dim, len(med_voc.word2idx)))
    
    def forward(self,history):
        visitTimes = len(history)
        diag_emb = []
        med_emb = []
        for times in range(visitTimes-1):
            diag = torch.cat(history[times][0])
            med = torch.cat(history[times][2])
            past_diag = torch.mean(self.diag_embedding(diag),axis = 0)
            past_med = torch.mean(self.med_embedding(med),axis = 0)
            diag_emb.append(past_diag)
            med_emb.append(past_med) 
        diag_emb = torch.stack(diag_emb).mean(dim=0)
        med_emb = torch.stack(med_emb).mean(dim=0)

        cur_diag = torch.cat(history[visitTimes-1][0])
        cur_diag_emb = torch.mean(self.diag_embedding(cur_diag),axis = 0)

        diag_emb = self.dense1(diag_emb)
        cur_diag_emb = self.dense1(cur_diag_emb)
        med_emb = self.dense2(med_emb)

        patient = torch.cat((diag_emb,cur_diag_emb,med_emb),axis = 0)
        med = torch.sigmoid(self.cls(patient))
        return torch.unsqueeze(med,dim=0)

