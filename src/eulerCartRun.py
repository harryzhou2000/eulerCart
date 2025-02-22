import torch
import eulerCart.gas as gas
import eulerCart.cart as cart
import eulerCart.solver as ecsolver
import numpy as np
import matplotlib.pyplot as plt
import time


def Run():
    sz = (4096, 4096)
    CFL = 0.2
    tOuts = [0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
    see = 10
    uBlock = [0.5, -1, 0, 4]

    solver = ecsolver.EulerCartSolver2D(device="cuda:1", dataType=torch.float32)
    solver.set_grid(sz, [0, 1], [0, 1])
    solver.set_gamma(1.4)
    solver.set_uBack([1, 0, 0, 2.5])
    xcm, ycm = solver.xcm, solver.ycm
    ifBlock = (
        (xcm < 0.75)
        .logical_and_(xcm > 0.25)
        .logical_and(ycm < 0.75)
        .logical_and(ycm > 0.25)
    )

    block = cart.sub2indFromBool_pytorch(ifBlock)

    u0 = solver.get_init_u_uniform([1, 0, 0, 2.5])
    u0.view(-1, 4)[block, :] = torch.tensor(uBlock, device=u0.device).reshape(1, 4)

    u = u0.clone()

    t = 0.0
    iOut = 0

    fig, ax = plt.subplots()
    fig.set_size_inches((12, 11))

    tic = time.perf_counter()
    iterFull = 0
    for tNext in tOuts:
        for iter in range(1, 10000000 + 1):
            uNew, tNew, ifOutS, info = solver.integrate_time(u, t, tNext, CFL, see)
            iterFull += info.nSteps
            u = uNew
            t = tNew
            ifOut = ifOutS

            rho = u[:, :, 0]

            # with plt.ion():
            # fig.clear()
            # fig.clf()
            ax.cla()
            ax.pcolormesh(xcm.cpu(), ycm.cpu(), rho.cpu(), shading="auto", cmap="jet")
            ax.axis("equal")
            plt.show(block=False)
            plt.pause(0.1)
            fig.savefig("out/cfig.png")

            print(
                "iter %d, t = [%.4e], dt = [%.4e], cpuTime = [%.2e]"
                % (iterFull, t, info.dt, (time.perf_counter() - tic))
            )
            # print(rho.max())
            tic = time.perf_counter()

            # if iterFull >= 100:
            #     return

            if ifOut:
                break

        assert ifOut, f"the solver failed to reach time {tNext}"

        rho = u[:, :, 0]
        figc, axc = plt.subplots()
        figc.set_size_inches((12, 11))
        axc.cla()
        axc.pcolormesh(xcm.cpu(), ycm.cpu(), rho.cpu(), shading="auto", cmap="jet")
        axc.axis("equal")
        axc.set_title("t = %g" % (t))
        figc.savefig("out/rho_t_%g.png" % (t))

        iOut += 1
        if iOut >= len(tOuts):
            break
        pass


if __name__ == '__main__':
    Run()
