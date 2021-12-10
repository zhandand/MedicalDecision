# -*- coding: utf-8 -*-
# @Time       : 2021/11/10 16:01:56
# @Author     : Zhan Genze <947783684@qq.com>
# @Project    : gzzhan
# @Description:  statistical model for medication recommendation
#                calculate p(m|d) according to the records
#                choose the highest p(m|d) for patients who got disease d

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PMD(nn.Module):
    def __init__(self, *args):
        super(PMD,self).__init__()
        vocab = args[1]['vocab']
        self.diag_voc = vocab['diag_voc']
        self.med_voc = vocab['med_voc']
        self.origin_MD = np.zeros((len(self.diag_voc.idx2word),len(self.med_voc.idx2word)))
        self.diag_times = np.zeros((len(self.diag_voc.idx2word)))

        self.dense = nn.Linear(16 * 16, 512)
    
    def forward(self,x):
        # due to batch, the x is [[]].
        mhot_diag, mhot_med = x
        mhot_diag, mhot_med = mhot_diag.squeeze().data.cpu().numpy().astype('int64'), mhot_med.squeeze().data.cpu().numpy()
        diag = np.where(mhot_diag==1)
        if self.training:
            # iterate matrix during training
            self.origin_MD[diag] += mhot_med
            self.diag_times += mhot_diag

        cur_diag_times = np.zeros((len(self.diag_voc.idx2word)))
        for i in range(len(self.diag_voc.idx2word)):
            if self.diag_times[i] !=0:
                cur_diag_times[i] = 1.0/self.diag_times[i]

        med_rec_matrix = np.matmul(np.diag(cur_diag_times), self.origin_MD)
        med_rec = med_rec_matrix[diag].sum(axis=0)
        med_rec = np.clip(med_rec,0,1)
        return  torch.unsqueeze(torch.tensor(med_rec,device='cuda'),dim=0)