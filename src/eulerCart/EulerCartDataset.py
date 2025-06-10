import torch

from torch.utils.data import Dataset, DataLoader

import numpy as np
import os


def count_dataset_files(dir):
    ds_base = os.path.abspath(dir)
    files = list(filter(lambda s: s.endswith(".safetensors"), os.listdir(ds_base)))
    return len(files)


class EulerCartTimeHistoryDataset(Dataset):

    def __init__(
        self,
        dir,
        pre_processor,
        device,
        nJumps=[1, 2, 4, 8],
        nTrans=4,
        min_i_file=0,
        max_i_file=100000000,
    ):
        ds_base = os.path.abspath(dir)
        files = list(filter(lambda s: s.endswith(".safetensors"), os.listdir(ds_base)))
        files = sorted(files)
        files = files[min_i_file:max_i_file]
        files = [os.path.join(ds_base, file) for file in files]
        self.files = files
        self.nfiles = len(files)

        file_pairs = []

        for nJump in nJumps:
            for iFile, file in enumerate(files):
                if iFile + nJump < len(files):
                    for itrans in range(nTrans):
                        file_pairs.append((files[iFile], files[iFile + nJump], itrans))
        self.file_pairs = file_pairs
        self.pre_processor = pre_processor
        self.device = device

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        from safetensors.torch import load_file

        file0, file1, itrans = self.file_pairs[idx]
        data0 = load_file(file0, device=self.device)
        data1 = load_file(file1, device=self.device)

        xcm = data0["xcm"]
        ycm = data0["ycm"]
        u0 = data0["u"]
        u1 = data1["u"]
        t0 = data0["t"]
        t1 = data1["t"]
        dt = t1 - t0

        return self.pre_processor(xcm, ycm, u0, u1, dt, itrans, self.device)


def EulerCart_SNT_pre_process(xcm, ycm, u0, u1, dt, itrans, device):

    Nx, Ny, dim_phy = u0.shape
    ge_in = torch.zeros((Nx, Ny, 5, 3), device=device)
    ge_in[:, :, 1, 0] = -1.0  # Le
    ge_in[:, :, 2, 0] = 1.0  # Ri
    ge_in[:, :, 3, 1] = -1.0  # Lo
    ge_in[:, :, 4, 1] = 1.0  # Up

    ge_in[0, :, 1, 2] = 1  # Le bnd
    ge_in[-1, :, 2, 2] = 1  # Ri bnd
    ge_in[:, 0, 3, 2] = 1  # Lo bnd
    ge_in[:, -1, 4, 2] = 1  # Up bnd

    assert 0 <= itrans < 4

    if itrans == 1 or itrans == 3:
        ge_in[:, :, :, 0] *= -1
        u0[:, :, 1] *= -1
        u1[:, :, 1] *= -1

    if itrans == 2 or itrans == 3:
        ge_in[:, :, :, 1] *= -1
        u0[:, :, 2] *= -1
        u1[:, :, 2] *= -1

    sample = torch.zeros((Nx, Ny, dim_phy + 1), device=device)
    sample[:, :, 0:dim_phy] = u0
    sample[..., dim_phy : dim_phy + 1] = dt
    label = torch.as_tensor(u1, device=device)

    return sample, ge_in, label


def EulerCart_SNT_F_MP(x: torch.Tensor):
    assert x.ndim == 4
    xLe = x.roll(1, 1)
    xRi = x.roll(-1, 1)
    xLo = x.roll(1, 2)
    xUp = x.roll(-1, 2)
    xLe[0, :] = 0
    xRi[-1, :] = 0
    xLo[:, 0] = 0
    xUp[:, -1] = 0
    xs = [x, xLe, xRi, xLo, xUp]
    xs = [t.unsqueeze(3) for t in xs]
    return torch.concat(xs, 3)
