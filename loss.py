import torch

def InfoNCE(similarity, t, index, coefficience):
    """CL loss

    Args:
        similarity ([type]): 
        t ([type]): temperature
    """
    similarity = similarity / t
    LogSoftmax = torch.nn.LogSoftmax(dim=1)
    Rows = torch.arange(similarity.shape[0])
    loss = -1.0* LogSoftmax(similarity)[Rows, index]
    return torch.mean(loss/coefficience)