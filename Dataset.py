# -*- coding: utf-8 -*-
# @Time       : 2021/11/10 16:54:46
# @Author     : Zhan Genze <947783684@qq.com>
# @Project    : gzzhan
# @Description:  根据不同的任务设置不同的dataset

import torch
from torch.utils.data import Dataset

from utils import id2multihot

torch.multiprocessing.set_sharing_strategy('file_system')


class MedRecommendDataset(Dataset):
    """ 药物推荐，根据历史记录及当前疾病（操作码）预测药物

    Args:
            Dataset ([type]): [description]
    """

    def __init__(self, records, vocab, single_records=None):
        def tranform(records):
            """ split each patient's visits into many single visit

            Args:
                    records ([type]): [description]
            """
            newRecords = []
            for patient in records:
                for visit in patient:
                    newRecords.append(visit)
            return newRecords
        if single_records == None:
            self.records = tranform(records)
        else:
            self.records = tranform(records) + tranform(single_records)
        self.vocab = vocab

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        visit = self.records[index]
        diag = id2multihot(visit[0], len(self.vocab["diag_voc"].idx2word))
        pro = id2multihot(visit[1], len(self.vocab["pro_voc"].idx2word))
        med = id2multihot(visit[2], len(self.vocab["med_voc"].idx2word))

        return (diag, med), med


class LongitudeMedRecDataset(Dataset):
    # 纵向药物推荐，根据t-1次疾病和药物以及当前疾病推荐药物
    def __init__(self, records, vocab):
        self.vocab = vocab
        self.newRecords = []
        for patient in records:
            for times in range(1, len(patient)):
                self.newRecords.append(patient[0:times+1])

    def __len__(self):
        return len(self.newRecords)

    def __getitem__(self, index):
        visit = self.newRecords[index]
        # history, med(multi-hot)
        return visit, id2multihot(visit[len(visit)-1][2], len(self.vocab["med_voc"].idx2word))

class MDCLDataset(Dataset):
    # 药物和疾病的正例索引
    def __init__(self, pairs):
        self.pairs = pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self,index):
        diag, med = self.pairs[index]
        return diag, med