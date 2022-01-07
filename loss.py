import torch
import torch.nn as nn


def InfoNCE(similarity, t, index, coefficience):
    """CL loss

    Args:
        similarity ([type]): 
        t ([type]): temperature
    """
    similarity = similarity / t
    LogSoftmax = torch.nn.LogSoftmax(dim=1)
    Rows = torch.arange(similarity.shape[0])
    loss = -1.0 * LogSoftmax(similarity)[Rows, index]
    return torch.mean(loss/coefficience)


def NCESoftmaxLoss(feats_q,feats_k):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""
    similarity = torch.matmul(feats_q, feats_k.t())
    bsz = similarity.shape[0]
    # positives on the diagonal
    label = torch.arange(bsz).cuda().long()
    criterion = nn.CrossEntropyLoss()
    loss = criterion(similarity, label)
    return loss
