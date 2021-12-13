# -*- coding: utf-8 -*-
# @Time       : 2021/11/08 21:40:40
# @Author     : Zhan Genze <947783684@qq.com>
# @Project    : gzzhan
# @Description:  pipeline for experiment

import pytorch_lightning as pl

from utils import load_sources, load_from_ckpt, read_config
from Callbacks import loadCallbacks
from DataModule import MedicalDataModule
from MedDec import MedDec
import torch

def train_valid_pipeline(**kwargs):
    sources = load_sources(kwargs['gpus'], **kwargs['add_sources'])
    if 'pretrain' in kwargs.keys():                # 加载预训练模型
        pretrain_configs = read_config(kwargs['pretrain']['config_path'])
        pretrain_model = MedDec(pretrain_configs,sources)
        load_from_ckpt(kwargs['pretrain']['save_path'], pretrain_model)
        kwargs['model']['pretrain'] = pretrain_model
    model = MedDec(kwargs,sources)                  # 加入额外的知识源
    callbacks = loadCallbacks(kwargs['Callbacks'])
    # turn validation before training off
    trainer = pl.Trainer(gpus = [kwargs['gpus']], auto_select_gpus=True,max_epochs = kwargs['epochs'],callbacks = list(callbacks.values()),num_sanity_val_steps=0,val_check_interval = 1.0)
    datamodule = MedicalDataModule(kwargs['data'],sources)
    trainer.fit(model, datamodule= datamodule)
    kwargs['trainer'] = trainer
    if "ModelCheckpoint" in callbacks.keys():
        kwargs['model']['save_path'] = callbacks['ModelCheckpoint'].best_model_path
    if 'pretrain' in kwargs.keys():
        del kwargs['model']['pretrain']

def test_pipeline(**kwargs):
    sources = load_sources(kwargs['gpus'], **kwargs['add_sources'])
    model = MedDec(kwargs,sources)
    #!: important line
    # checkpoint = torch.load(kwargs['model']['save_path'], map_location=lambda storage, loc: storage)
    # model.load_state_dict(checkpoint['state_dict'])
    load_from_ckpt(kwargs['model']['save_path'], model)
    datamodule = MedicalDataModule(kwargs['data'],sources)
    trainer = pl.Trainer(gpus =[kwargs['gpus']], max_epochs = kwargs['epochs'])
    trainer.test(model, datamodule= datamodule)

    