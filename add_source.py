# -*- coding: utf-8 -*-
# @Time       : 2021/11/10 20:26:24
# @Author     : Zhan Genze <947783684@qq.com>
# @Project    : gzzhan
# @Description:  additional resources
import os

from utils import read_pkl


def load_sources(configs):
    assert len(configs['type']) == len(configs['data_path'])
    sources = {}
    for item in zip(configs['type'], configs['data_path']):
        if os.path.splitext(item[1])[-1] == '.pkl':
            sources[item[0]] = read_pkl(item[1])
        elif os.path.splitext(item[1])[-1] == '.npy':
            sources[item[0]] = read_npy(item[1])
    print("Load additional sources done...")
    return sources
