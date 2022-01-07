# -*- coding: utf-8 -*-
# @Time       : 2021/11/08 21:41:09
# @Author     : Zhan Genze <947783684@qq.com>
# @Project    : gzzhan
# @Description:  dataloader

import os
import numpy as np
import pytorch_lightning as pl
from utils import split, read_pkl, read_npy, graph_batcher
from Dataset import MedRecommendDataset, LongitudeMedRecDataset, MDCLDataset, GCCCLDataset
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


class MedicalDataModule(pl.LightningDataModule):
    def __init__(self, *args):
        super().__init__()
        self.kwargs = args[0]
        self.resource = args[1]
        self.data = self.getData(self.kwargs['path'])
        self.train_ratio = self.kwargs['split']['train_ratio']
        self.test_ratio = self.kwargs['split']['test_ratio']
        self.valid_ratio = self.kwargs['split']['valid_ratio']
        self.target = self.kwargs['target']
        self.shuffle = self.kwargs['shuffle']
        self.seed = self.kwargs['seed']
        self.num_workers = self.kwargs['num_workers']
        self.batch_size = self.kwargs['batch_size']
        print("Load dataset done...")

    def prepare_data(self):
        """
        根据不同的任务，即target值修改dataset组织方式
        """
        if self.target in ["med_recommend", 'longitude_med_rec']:
            self.train_dateset, self.valid_dataset, self.test_dataset = split(
                self.data, self.train_ratio, self.test_ratio, self.valid_ratio, self.seed, self.shuffle)
        if self.target == "med_recommend":
            if "single_records" not in self.resource.keys():
                self.resource['single_records'] = None
            self.train_dateset = MedRecommendDataset(
                self.train_dateset, self.resource['vocab'], self.resource['single_records'])
            self.valid_dataset = MedRecommendDataset(
                self.valid_dataset, self.resource['vocab'])
            self.test_dataset = MedRecommendDataset(
                self.test_dataset, self.resource['vocab'])
        elif self.target == 'longitude_med_rec':
            self.train_dateset = LongitudeMedRecDataset(
                self.train_dateset, self.resource['vocab'])
            self.valid_dataset = LongitudeMedRecDataset(
                self.valid_dataset, self.resource['vocab'])
            self.test_dataset = LongitudeMedRecDataset(
                self.test_dataset, self.resource['vocab'])
        elif self.target == 'cl_pretrain':
            # self.train_dateset = MDCLDataset(self.train_dateset)
            # self.valid_dataset = MDCLDataset(self.valid_dataset)
            # self.test_dataset = MDCLDataset(self.test_dataset)
            self.train_dateset = GCCCLDataset(self.data, **self.kwargs['gcc'])
            self.valid_dataset = None
            self.test_dataset = None

    def train_dataloader(self):
        if self.target == "cl_pretrain":
            return DataLoader(self.train_dateset, sampler=SequentialSampler(self.train_dateset), batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, collate_fn=graph_batcher())
        else:
            return DataLoader(self.train_dateset, sampler=SequentialSampler(self.train_dateset), batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, sampler=SequentialSampler(self.valid_dataset), batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, sampler=SequentialSampler(self.test_dataset), batch_size=self.batch_size, num_workers=self.num_workers)

    def getData(self, path):
        if os.path.splitext(path)[-1] == '.pkl':
            return read_pkl(path)
        elif os.path.splitext(path)[-1] == '.npy':
            return read_npy(path)