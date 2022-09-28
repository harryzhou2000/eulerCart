import torch


def Le(d: torch.Tensor):
    return d.roll(1, 0)


def Ri(d: torch.Tensor):
    return d.roll(-1, 0)


def Lo(d: torch.Tensor):
    return d.roll(1, 1)


def Up(d: torch.Tensor):
    return d.roll(-1, 1)

def sub2indFromBool(r, s, sz, ifx):
    return r[ifx] * sz[0] + s[ifx]
