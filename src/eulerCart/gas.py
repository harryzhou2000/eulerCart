# import imp
import torch
from .cart import *
from . import tvd


def RoeSolver2(uLin: torch.Tensor, uRin: torch.Tensor, gammaIn: torch.Tensor):
    uL = uLin.view(-1, 4)
    uR = uRin.view(-1, 4)
    gamma = gammaIn.view(-1, 1)

    rhoL, UxL, UyL, UsqrL, EL, pL, aL, HL = F_uExpand_V1(uL, gamma)
    rhoR, UxR, UyR, UsqrR, ER, pR, aR, HR = F_uExpand_V1(uR, gamma)

    rhoLsqrt = rhoL.sqrt()
    rhoRsqrt = rhoR.sqrt()
    rhoLRsqrtSum = rhoLsqrt + rhoRsqrt
    UxRoe = (rhoLsqrt * UxL + rhoRsqrt * UxR) / rhoLRsqrtSum
    UyRoe = (rhoLsqrt * UyL + rhoRsqrt * UyR) / rhoLRsqrtSum
    HRoe = (rhoLsqrt * HL + rhoRsqrt * HR) / rhoLRsqrtSum
    UsqrRoe = UxRoe**2 + UyRoe**2
    asqrRoe = (gamma - 1) * (HRoe - 0.5 * UsqrRoe)
    aRoe = asqrRoe.sqrt()

    L0Roe = UxRoe - aRoe
    LmRoe = UxRoe
    L4Roe = UxRoe + aRoe

    deltaEF = 0.2 * (UsqrRoe.sqrt() + aRoe)
    L0Fix = L0Roe.abs() < deltaEF
    L4Fix = L4Roe.abs() < deltaEF
    L0Roe[L0Fix] = (
        0.5
        * torch.sign(L0Roe[L0Fix])
        * (L0Roe[L0Fix] ** 2 + deltaEF[L0Fix] ** 2)
        / deltaEF[L0Fix]
    )
    L4Roe[L4Fix] = (
        0.5
        * torch.sign(L4Roe[L4Fix])
        * (L4Roe[L4Fix] ** 2 + deltaEF[L4Fix] ** 2)
        / deltaEF[L4Fix]
    )

    rev0 = torch.ones_like(uL)
    rev1 = rev0.clone()
    rev2 = rev0.clone()
    rev4 = rev0.clone()

    # rev0[:, [0]] = 1
    rev0[:, [1]] = L0Roe
    rev0[:, [2]] = UyRoe
    rev0[:, [3]] = HRoe - UxRoe * aRoe

    # rev1[:, [0]] = 1
    rev1[:, [1]] = UxRoe
    rev1[:, [2]] = UyRoe
    rev1[:, [3]] = 0.5 * UsqrRoe

    rev2[:, [0]] = 0
    rev2[:, [1]] = 0
    # rev2[:, [2]] = 1
    rev2[:, [3]] = UyRoe

    # rev4[:, 0[]] = 1
    rev4[:, [1]] = L4Roe
    rev4[:, [2]] = UyRoe
    rev4[:, [3]] = HRoe + UxRoe * aRoe

    rhoInc = rhoR - rhoL
    pinc = pR - pL
    rhoRoe = rhoLsqrt * rhoRsqrt
    incUx = UxR - UxL
    incUy = UyR - UyL

    alpha0 = 0.5 * (pinc - rhoRoe * incUx * aRoe) / asqrRoe
    alpha1 = rhoInc - pinc / asqrRoe
    alpha2 = rhoRoe * incUy
    alpha4 = 0.5 * (pinc + rhoRoe * incUx * aRoe) / asqrRoe

    # uinc = uR - uL
    # alpha2 = uinc[:, [2]] - UyRoe * rhoInc
    # incU4c = uinc[:, [3]] - (uinc[:, [2]] - UyRoe * rhoInc) * UyRoe
    # alpha1 = (gamma-1) / asqrRoe * (rhoInc *
    #                                 (HRoe - UxRoe**2) + UxRoe * uinc[:, [1]] - incU4c)
    # alpha0 = 0.5 * (rhoInc * L4Roe - uinc[:, [1]] - aRoe * alpha1) / aRoe
    # alpha4 = rhoInc - (alpha0 + alpha1)

    FL = F_u2F2(uL, UxL, pL, gamma)
    FR = F_u2F2(uR, UxR, pR, gamma)
    F = (
        0.5 * (FL + FR)
        - 0.5 * alpha0 * L0Roe.abs() * rev0
        - 0.5 * alpha1 * LmRoe.abs() * rev1
        - 0.5 * alpha2 * LmRoe.abs() * rev2
        - 0.5 * alpha4 * L0Roe.abs() * rev4
    )

    return F.view_as(uLin)


def F_uExpand_V1(uIn: torch.Tensor, gammaIn: torch.Tensor):
    rho = uIn[:, [0]]  # ! rho is a reference here
    Ux = uIn[:, [1]] / rho
    Uy = uIn[:, [2]] / rho
    E = uIn[:, [3]] / rho
    Usqr = Ux**2 + Uy**2
    p = (E - 0.5 * Usqr) * rho * (gammaIn - 1)
    a = torch.sqrt(gammaIn * p / rho)
    H = E + p / rho
    return (rho, Ux, Uy, Usqr, E, p, a, H)


def F_FixU(uIn: torch.Tensor, gammaIn: torch.Tensor):
    uL = uIn.view(-1, 4)
    gamma = gammaIn.view(-1, 1)
    rho = uL[:, 0]
    Ux = uL[:, 1] / rho
    Uy = uL[:, 2] / rho
    E = uL[:, 3] / rho
    Usqr = Ux**2 + Uy**2
    p = (E - 0.5 * Usqr) * rho * (gamma - 1)
    pMax = p.max()
    rMax = rho.max()
    pThres = pMax * 1e-3
    rThres = rMax * 1e-3

    pSet = p < pThres
    rSet = rho < rThres
    rho[rSet] = rThres
    p[pSet] = pThres

    uL[:, 0] = rho
    uL[:, 1] = Ux * rho
    uL[:, 2] = Uy * rho
    uL[:, 3] = p / (gamma - 1) + 0.5 * Usqr * rho


def F_u2F2(uIn: torch.Tensor, ux: torch.Tensor, p: torch.Tensor, gammaIn: torch.Tensor):
    F = torch.zeros_like(uIn)
    F[:, 0] = uIn[:, 1]
    F[:, 1] = uIn[:, 1] * ux.view(-1) + p.view(-1)
    F[:, 2] = uIn[:, 2] * ux.view(-1)
    F[:, 3] = (uIn[:, 3] + p.view(-1)) * ux.view(-1)
    return F


def F_u_yfce2xfce(U: torch.Tensor):
    Uc = U.view(-1, 4)
    ut = Uc[:, 1].clone()
    Uc[:, 1] = -Uc[:, 2]
    Uc[:, 2] = ut


def F_u_xfce2yfce(U: torch.Tensor):
    Uc = U.view(-1, 4)
    ut = Uc[:, 1].clone()  # ! easy to omit!!
    Uc[:, 1] = Uc[:, 2]
    Uc[:, 2] = -ut


def F_u2maxLamX(uIn: torch.Tensor, gammaIn: torch.Tensor):
    u = uIn.view(-1, 4)
    gamma = gammaIn.view(-1, 1)
    rho = u[:, 0]
    Ux = u[:, 1] / rho
    Uy = u[:, 2] / rho
    Usqr = Ux**2 + Uy**2

    p = (gamma - 1) * (u[:, 3] - 0.5 * Usqr * rho)

    a = (gamma * p / rho).sqrt()
    um = Ux - a
    up = Ux + a
    return um.abs().maximum(up.abs()).view_as(uIn[:, :, 0])


def EulerCartRHS(
    uIn,
    hLe,
    hRi,
    hLo,
    hUp,
    hx,
    hy,
    Vol,
    wBLe,
    wBRi,
    wBLo,
    wBUp,
    uLeB,
    uRiB,
    uLoB,
    uUpB,
    gamma,
):
    # u = uIn.clone()
    u = uIn
    DLe = (u - Le(u)) / hLe.unsqueeze(2)
    DRi = (Ri(u) - u) / hRi.unsqueeze(2)
    DUp = (Up(u) - u) / hUp.unsqueeze(2)
    DLo = (u - Lo(u)) / hLo.unsqueeze(2)

    DLe.view(-1, 4)[wBLe, :] = 0
    DRi.view(-1, 4)[wBRi, :] = 0
    DUp.view(-1, 4)[wBUp, :] = 0
    DLo.view(-1, 4)[wBLo, :] = 0

    DCx = tvd.F_TVD_Slope(DLe, DRi)
    DCy = tvd.F_TVD_Slope(DLo, DUp)
    # print(DCx.max())

    uLe = -hx.unsqueeze(2) * 0.5 * DCx + u
    uRi = +hx.unsqueeze(2) * 0.5 * DCx + u
    uLo = -hy.unsqueeze(2) * 0.5 * DCy + u
    uUp = +hy.unsqueeze(2) * 0.5 * DCy + u

    FLe_Ri = uLe
    FLe_Le = Le(uRi)
    FLe_Le[0, :, :] = uLeB.view(1, 4)
    FLe_Le.view(-1, 4)[wBLe, :] = FLe_Ri.view(-1, 4)[wBLe, :]
    FLe_Le.view(-1, 4)[wBLe, 1] = -FLe_Ri.view(-1, 4)[wBLe, 1]

    FRi_Le = uRi
    FRi_Ri = Ri(uLe)
    FRi_Ri[-1, :, :] = uRiB.view(1, 4)
    FRi_Ri.view(-1, 4)[wBRi, :] = FRi_Le.view(-1, 4)[wBRi, :]
    FRi_Ri.view(-1, 4)[wBRi, 1] = -FRi_Le.view(-1, 4)[wBRi, 1]

    FLo_Up = uLo
    FLo_Lo = Lo(uUp)
    FLo_Lo[:, 0, :] = uLoB.view(1, 4)
    FLo_Lo.view(-1, 4)[wBLo, :] = FLo_Up.view(-1, 4)[wBLo, :]
    FLo_Lo.view(-1, 4)[wBLo, 2] = -FLo_Up.view(-1, 4)[wBLo, 2]

    FUp_Lo = uUp
    FUp_Up = Up(uLo)
    FUp_Up[:, -1, :] = uUpB.view(1, 4)
    FUp_Up.view(-1, 4)[wBUp, :] = FUp_Lo.view(-1, 4)[wBUp, :]
    FUp_Up.view(-1, 4)[wBUp, 2] = -FUp_Lo.view(-1, 4)[wBUp, 2]

    F_FixU(FLe_Le, gamma)
    F_FixU(FLe_Ri, gamma)
    F_FixU(FRi_Le, gamma)
    F_FixU(FRi_Ri, gamma)
    F_FixU(FLo_Lo, gamma)
    F_FixU(FLo_Up, gamma)
    F_FixU(FUp_Lo, gamma)
    F_FixU(FUp_Up, gamma)

    fLe = RoeSolver2(FLe_Le, FLe_Ri, gamma)
    fRi = RoeSolver2(FRi_Le, FRi_Ri, gamma)
    F_u_xfce2yfce(FUp_Lo)
    F_u_xfce2yfce(FUp_Up)
    F_u_xfce2yfce(FLo_Lo)
    F_u_xfce2yfce(FLo_Up)
    fUp = RoeSolver2(FUp_Lo, FUp_Up, gamma)
    fLo = RoeSolver2(FLo_Lo, FLo_Up, gamma)
    F_u_yfce2xfce(fUp)
    F_u_yfce2xfce(fLo)

    dUdt = (
        (fLe - fRi) * hy.unsqueeze(2) + (fLo - fUp) * hx.unsqueeze(2)
    ) / Vol.unsqueeze(2)

    return dUdt
