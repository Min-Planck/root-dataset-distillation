import math 
import numpy as np 
import torch 

def kl_divergence(p, q):
    p = np.array(p)
    q = np.array(q)

    mask = (p != 0) & (q != 0)
    p = p[mask]
    q = q[mask]

    return np.sum(p * np.log(p / q))

def jensen_shannon_divergence_distance(p, q):
    p = np.array(p)
    q = np.array(q)

    m = (p + q) / 2

    kl_p_m = kl_divergence(p, m)
    kl_q_m = kl_divergence(q, m)

    return (kl_p_m + kl_q_m) / 2

def hellinger(p, q):
    """
    Hellinger distance between two discrete distributions.
    """
    list_of_squares = []
    for p_i, q_i in zip(p, q):
        s = (math.sqrt(p_i) - math.sqrt(q_i)) ** 2
        list_of_squares.append(s)

    sosq = sum(list_of_squares)    

    return math.sqrt(sosq) / math.sqrt(2)

def gradient_distance(grad1, grad2, device): 
    dist = torch.tensor(0.0).to(device)
    for gr, gs in zip(grad1, grad2):
        shape=gr.shape
        if len(shape) == 4:
            gr = gr.reshape(shape[0], shape[1] * shape[2] * shape[3])
            gs = gs.reshape(shape[0], shape[1] * shape[2] * shape[3])
        elif len(shape) == 3:
            gr = gr.reshape(shape[0], shape[1] * shape[2])
            gs = gs.reshape(shape[0], shape[1] * shape[2])
        elif len(shape) == 2:
            tmp = 'do nothing'
        elif len(shape) == 1:
            gr = gr.reshape(1, shape[0])
            gs = gs.reshape(1, shape[0])
            continue
        dis_weight = torch.sum(1 - torch.sum(gr * gs, dim=-1) / (torch.norm(gr, dim=-1) * torch.norm(gs, dim=-1)+ 0.000001))
        dist += dis_weight
    return dist