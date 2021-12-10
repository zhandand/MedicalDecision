import os
import random
from typing import List

import torch
import dill
import numpy as np
import yaml
from sklearn.model_selection import train_test_split


def read_config(config_path):
    """read yaml file from config path

    Args:
            config_path ([str]): config file path

    Raises:
            FileNotFoundError: file not exist

    Returns:
            [dict]: config setting for experiment
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return yaml.load(content, Loader=yaml.FullLoader)


def save_config(config_path, config):
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)


def read_pkl(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    return dill.load(open(file_path, 'rb'))


def read_npy(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    return np.load(open(file_path, 'rb'))


def mk_root(root_dir):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)


def mk_rundir(root_dir, run_name):
    root_dir = os.getcwd() + root_dir
    mk_root(root_dir)
    run_path = os.path.join(root_dir, run_name)
    if not os.path.exists(run_path):
        os.mkdir(run_path)

    old_rundirs = [file for file in os.listdir(
        run_path) if os.path.isdir(os.path.join(run_path, file))]
    if(old_rundirs == []):
        cur_path = os.path.join(run_path, "0")
    else:
        cur_path = max([int(x) for x in old_rundirs])+1
    os.mkdir(os.path.join(run_path, str(cur_path)))
    return os.path.join(run_path, str(cur_path))


def split(dataset, train_ratio, test_ratio, valid_ratio, seed=31, shuffle=True):
    """split train, valid, test dataset

    Args:
            dataset ([type]): [description]
            train_ratio ([type]): [description]
            test_ratio ([type]): [description]
            seed ([type]): [description]
            shuffle (bool, optional): [description]. Defaults to True.

    Returns:
            [type]: [description]
    """
    assert train_ratio + test_ratio <= 1

    # train_set, test_set, valid_set = np.split(
    #     dataset, [int(train_ratio * len(dataset)), int((train_ratio + test_ratio) * len(dataset))])
    train_set, test_set = train_test_split(
        dataset, test_size=1 - train_ratio, random_state=seed, shuffle=shuffle)
    test_set, valid_set = train_test_split(
        test_set, test_size=valid_ratio/(test_ratio+valid_ratio), shuffle=False)
    return train_set, valid_set, test_set


# def load_Dataloader(train_dataset, test_dataset, valid_dataset=None):
#     if valid_dataset is None:
# 	    return DataLoader(train_dataset, sampler=SequentialSampler(train_dataset), batch_size=1), DataLoader(test_dataset, sampler=SequentialSampler(
#         test_dataset), batch_size=1)
#     else:
#         return DataLoader(train_dataset, sampler=SequentialSampler(train_dataset), batch_size=1),DataLoader(test_dataset, sampler=SequentialSampler(
#             test_dataset), batch_size=1),DataLoader(valid_dataset, sampler=SequentialSampler(
#         valid_dataset), batch_size=1)


def id2multihot(ids: List[int], vocab_size: int):
    """
    change list of medical codes to multihot vector
    Args:
            ids (List): existing medical codes
            vocab_size (int): [description]

    Returns:
            [type]: multihot vector
    """
    multi_hot = np.zeros(vocab_size)
    multi_hot[ids] = 1
    return multi_hot


def load_sources(gpu,**configs):
    assert len(configs['type']) == len(configs['data_path'])
    sources = {}
    for item in zip(configs['type'], configs['data_path']):
        if os.path.splitext(item[1])[-1] == '.pkl':
            sources[item[0]] = read_pkl(item[1])
        elif os.path.splitext(item[1])[-1] == '.npy':
            sources[item[0]] = to_device(read_npy(item[1]),gpu)
    print("Load additional sources done...")
    return sources


def to_device(data, gpu):
    if isinstance(data,np.ndarray):
        data = torch.from_numpy(data)
        device = torch.device("cuda:"+str(gpu))
        return data.to(device)
    else:
        raise NotImplementedError()


def load_from_ckpt(ckpt_path, model):
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])