import torch


def F_TVD_Slope(dL: torch.Tensor, dR: torch.Tensor):
    d = (dL.sign() + dR.sign()) * (dL*dR).abs() / ((dL+dR).abs() + 1e-8)
    return d
