import torch
import matplotlib.pyplot as plt
import time
import json

from . import gas as gas
from . import cart as cart


class EulerCartSolver2D:
    def __init__(self, device="cuda", dataType=torch.float32):
        self.dataType = dataType
        self.device = device

    def __gen_grid(self):
        sz = (self.xs.shape[0] - 1, self.ys.shape[0] - 1)
        self.xc = 0.5 * (self.xs[:-1] + self.xs[1:])
        self.yc = 0.5 * (self.ys[:-1] + self.ys[1:])

        self.hx = self.xs[1:] - self.xs[:-1]
        self.hy = self.ys[1:] - self.ys[:-1]

        self.hxm, self.hym = torch.meshgrid(self.hx, self.hy, indexing="ij")
        self.xcm, self.ycm = torch.meshgrid(self.xc, self.yc, indexing="ij")
        self.vol = self.hxm * self.hym

        self.hLe = self.xcm - cart.Le(self.xcm)
        self.hRi = cart.Ri(self.xcm) - self.xcm
        self.hLo = self.ycm - cart.Lo(self.ycm)
        self.hUp = cart.Up(self.ycm) - self.ycm
        self.hLe[0, :] = self.hLe[1, :]
        self.hRi[-1, :] = self.hLe[-2, :]
        self.hLo[:, 0] = self.hLo[:, 1]
        self.hUp[:, -1] = self.hUp[:, -2]

        rs = torch.tensor(range(sz[0]), device=self.device, dtype=torch.long)
        ss = torch.tensor(range(sz[1]), device=self.device, dtype=torch.long)
        rsm, ssm = torch.meshgrid(rs, ss, indexing="ij")

        ifwBLe = rsm == 0
        ifwBRi = rsm == sz[0] - 1
        ifwBLo = ssm == 0
        ifwBUp = ssm == sz[1] - 1

        self.wBLe = cart.sub2indFromBool_pytorch(ifwBLe)
        self.wBRi = cart.sub2indFromBool_pytorch(ifwBRi)
        self.wBLo = cart.sub2indFromBool_pytorch(ifwBLo)
        self.wBUp = cart.sub2indFromBool_pytorch(ifwBUp)

    def set_grid(self, sz: tuple, xlim: tuple, ylim: tuple):
        self.xs = torch.linspace(
            xlim[0], xlim[1], sz[0] + 1, device=self.device, dtype=self.dataType
        )
        self.ys = torch.linspace(
            ylim[0], ylim[1], sz[1] + 1, device=self.device, dtype=self.dataType
        )
        self.__gen_grid()

    def set_gamma(self, gamma: float):
        self.gamma = torch.tensor([gamma], device=self.device, dtype=self.dataType)

    def set_uBack(self, uBack: list):
        self.uBack = torch.tensor(uBack, device=self.device, dtype=self.dataType)

    def __getstate__(self):
        # Return a dictionary containing the state of the object
        return {
            "xs": self.xs,
            "ys": self.ys,
            "uBack": self.uBack,
            "gamma": self.gamma,
        }

    def __setstate__(self, state):
        # Restore the state of the object from the dictionary
        self.xs = state["xs"]
        self.ys = state["ys"]
        self.uBack = state["uBack"]
        self.gamma = state["gamma"]
        self.__gen_grid()

    def get_init_u_uniform(self, uInit: list):
        uInitT = torch.tensor(uInit, device=self.device, dtype=self.dataType)
        sz = self.vol.shape
        u0 = torch.zeros((sz[0], sz[1], 4), device=self.device, dtype=self.dataType)
        u0[:, :, :] = uInitT.reshape(1, 1, 4)
        return u0

    def integrate_time(
        self,
        u: torch.Tensor,
        t0: float,
        t1: float,
        CFLf: float = 0.2,
        iter_max: int = 10000000,
    ) -> tuple[torch.Tensor, float, bool, any]:
        CFL = torch.tensor([CFLf], device=self.device, dtype=self.dataType)
        u = u.clone()  # avoid mutating

        t = t0
        for iter in range(1, iter_max + 1):
            UxMax = gas.F_u2maxLamX(u, self.gamma)
            uy = u.clone()
            gas.F_u_xfce2yfce(uy)
            UyMax = gas.F_u2maxLamX(uy, self.gamma)
            dtx = self.hxm / UxMax
            dty = self.hym / UyMax
            dtCFL = dtx.minimum(dty).min() * CFL
            dt = dtCFL
            ifOut = False
            if t + dt >= t1:
                ifOut = True
                dt = t1 - t

            def getRHS(uc):
                return gas.EulerCartRHS(
                    uc,
                    self.hLe,
                    self.hRi,
                    self.hLo,
                    self.hUp,
                    self.hxm,
                    self.hym,
                    self.vol,
                    self.wBLe,
                    self.wBRi,
                    self.wBLo,
                    self.wBUp,
                    self.uBack,
                    self.uBack,
                    self.uBack,
                    self.uBack,
                    self.gamma,
                )

            dudt0 = getRHS(u)
            u1 = u + dt * dudt0
            gas.F_FixU(u1, self.gamma)
            dudt1 = getRHS(u1)
            unew = 0.5 * u + 0.5 * u1 + 0.5 * dt * dudt1
            gas.F_FixU(unew, self.gamma)
            u = unew.clone()

            t = t + dt

            if ifOut:
                t = t1
                break

        class Info:
            def __getstate__(self):
                # Return a dictionary containing the state of the object
                return {
                    "dt": float(self.dt),
                    "dtCFL": float(self.dtCFL),
                    "t": float(self.t),
                    "nSteps": int(self.nSteps),
                }

            def __setstate__(self, state):
                # Restore the state of the object from the dictionary
                self.dt = state["dt"]
                self.dtCFL = state["dtCFL"]
                self.t = state["t"]
                self.nSteps = state["nSteps"]

            def toJson(self):
                return self.__getstate__()

        info = Info()
        info.dt = float(dt)
        info.dtCFL = float(dtCFL)
        info.nSteps = int(iter)
        info.t = float(t)

        return u, t, ifOut, info


# a new type:
