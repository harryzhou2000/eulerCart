import torch
import gas
import cart


if __name__ == '__main__':
    sz = (5, 5)
    device = 'cpu'
    CFL = torch.Tensor([0.2], device=device)
    tOuts = [0, 0.1]

    gamma = torch.Tensor([1.4], device=device)
    uBack = torch.Tensor([1, 0, 0, 2.5], device=device)
    uBlock = torch.Tensor([0.5, -1, 0, 4], device=device)

    xs = torch.linspace(0, 1, sz[0] + 1, device=device)
    ys = torch.linspace(0, 1, sz[1] + 1, device=device)

    rs = torch.tensor(range(sz[0]), device=device, dtype=torch.long)
    ss = torch.tensor(range(sz[1]), device=device, dtype=torch.long)

    rsm, ssm = torch.meshgrid(rs, ss, indexing='ij')

    xc = 0.5 * (xs[:-1] + xs[1:])
    yc = 0.5 * (ys[:-1] + ys[1:])

    hx = xs[1:] - xs[:-1]
    hy = ys[1:] - ys[:-1]

    hxm, hym = torch.meshgrid(hx, hy, indexing='ij')
    xcm, ycm = torch.meshgrid(xc, yc, indexing='ij')

    hLe = hxm - cart.Le(hxm)
    hRi = cart.Ri(hxm) - hxm
    hLo = hym - cart.Lo(hym)
    hUp = cart.Up(hym) - hym

    vol = hxm * hym

    ifwBLe = rsm == 0
    ifwBRi = rsm == sz[0] - 1
    ifwBLo = ssm == 0
    ifwBUp = ssm == sz[1] - 1
    ifBlock = (xcm < 0.75).logical_and_(xcm > 0.25).logical_and(
        ycm < 0.75).logical_and(ycm > 0.25)

    wBLe = cart.sub2indFromBool(rsm, ssm, sz, ifwBLe)
    wBRi = cart.sub2indFromBool(rsm, ssm, sz, ifwBRi)
    wBLo = cart.sub2indFromBool(rsm, ssm, sz, ifwBLo)
    wBUp = cart.sub2indFromBool(rsm, ssm, sz, ifwBUp)

    block = cart.sub2indFromBool(rsm, ssm, sz, ifBlock)

    u0 = torch.zeros((sz[0], sz[1], 4), device=device)

    u0.view(-1, 4)[:, :] = uBack.view(1, 4)
    u0.view(-1, 4)[block, :] = uBlock.view(1, 4)

    # print(xcm)
    # print(ycm)
    # print(u0[:, :, 0])

    u = u0