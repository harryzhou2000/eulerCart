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
    """
    Converts boolean indices to linear indices for 2D matrices.

    Parameters:
        r (torch.Tensor): The row indices tensor.
        s (torch.Tensor): The column indices tensor.
        sz (tuple): A tuple containing the size of the original matrix.
                   Format: (number of rows, number of columns)
        ifx (torch.Tensor): A boolean mask indicating valid indices.

    Returns:
        torch.Tensor: Linear indices corresponding to the True values in the boolean mask.
    """
    # Check if all tensors have the same shape
    assert r.shape == s.shape == ifx.shape, "r, s, and ifx must have the same shape"

    # Ensure that all are 2-dimensional
    assert (
        len(r.shape) == len(s.shape) == len(ifx.shape) == 2
    ), "r, s, and ifx must be 2-dimensional tensors"

    return r[ifx] * sz[0] + s[ifx]


def sub2indFromBool_pytorch(ifx: torch.Tensor):
    sz = torch.tensor(ifx.shape, device=ifx.device, dtype=torch.long)
    ldi = torch.cumprod(sz, dim=0).flip(0) / sz[-1]
    subs = torch.nonzero(ifx)

    return torch.sum(subs * ldi.reshape(1, -1), dim=1, dtype=torch.long)
