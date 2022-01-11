import os
from typing import List

import dgl
import dill
import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split

from MedDec import MedDec


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
    multi_hot = np.zeros(vocab_size, dtype=float)
    multi_hot[ids] = 1.0
    return torch.tensor(multi_hot.astype('float32'))


def load_sources(gpu, **configs):
    assert len(configs['type']) == len(configs['data_path'])
    sources = {}
    for item in zip(configs['type'], configs['data_path']):
        if os.path.splitext(item[1])[-1] == '.pkl':
            sources[item[0]] = read_pkl(item[1])
        elif os.path.splitext(item[1])[-1] == '.npy':
            sources[item[0]] = to_device(read_npy(item[1]), gpu)
    print("Load additional sources done...")
    return sources


def to_device(data, gpu):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
        device = torch.device("cuda:"+str(gpu))
        return data.to(device)
    else:
        raise NotImplementedError()


def load_from_ckpt(ckpt_path, model):
    checkpoint = torch.load(
        ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])


def load_pretrain(*args):
    # 加载预训练模型
    kwargs = args[0]
    sources = args[1]
    pretrain_configs = read_config(kwargs['config_path'])
    pretrain_model = MedDec(pretrain_configs, sources)
    load_from_ckpt(kwargs['save_path'], pretrain_model)
    return pretrain_model


def graph_batcher(batch):
    # 为gcc的batch到训练部分前的操作
    graph_q, graph_k = zip(*batch)
    graph_q, graph_k = dgl.batch(graph_q), dgl.batch(graph_k)
    return graph_q, graph_k


def history_graph_batcher(batch):
    visit_batch = []
    target_batch = []
    for patient in batch:
        history, target = patient
        visit_in_graph = []
        for visit in history:
            visit_in_graph.append(
                [dgl.batch(visit[0]), visit[1].to("cuda:0"), visit[2].to("cuda:0")])
        visit_batch.append(visit_in_graph)
        target_batch.append(target)
    return visit_batch, torch.stack(target_batch, axis=0)


def collate_fn_distributor(type: str):
    if type == "cl_pretrain":
        return graph_batcher
    elif type == "g_longitude_med_rec":
        return history_graph_batcher
    else:
        return None


def random_walk(g, nodes, hops, restart_prob):
    """ 随机游走得到子图

    Args:
        g ([type]): 待采样的图
        nodes ([type]): 起始节点
        hops ([type]): 采样最远距离
        restart_prob ([type]): 重开概率

    Returns:
        [type]: 采样得到的子图
    """
    traces, types = dgl.sampling.random_walk(
        g, nodes, length=hops, restart_prob=restart_prob)
    concat_vids, concat_types, lengths, offsets = dgl.sampling.pack_traces(
        traces, types)
    vids = concat_vids.split(lengths.tolist())
    subgraphs = []
    for vid in vids:
        if len(vid == 1):
            # 确保子图至少包含两个节点
            subgraphs.append(random_walk(g, vid, hops, restart_prob)[0])
        else:
            subgraphs.append(g.subgraph(vid))
    return subgraphs
