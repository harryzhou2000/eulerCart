import torch
import eulerCart.gas as gas
import eulerCart.cart as cart
import eulerCart.solver as ecsolver
import numpy as np
import matplotlib.pyplot as plt
import time
import os, sys
import pickle, json

from safetensors.torch import save_file


def Run():
    # sz = (1024, 1024)
    sz = (10, 10)
    CFL = 0.2
    # tOuts = [0.001, 0.2, 0.25, 0.3, 0.4, 0.5]
    tOuts = [0.5]
    see = 1
    uBlock = [0.5, -1, 0, 4]
    outDir = "/home/harry/ssd1/data/eulerCart/box-test"
    outAuxDir = "out-test"
    save_figs = False

    os.makedirs(outDir, exist_ok=True)
    os.makedirs(outAuxDir, exist_ok=True)

    solver = ecsolver.EulerCartSolver2D(device="cuda", dataType=torch.float32)
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

    with open(os.path.join(outDir, "problem.pt"), "wb") as f:
        pickle.dump({"solver": solver, "u0": u0}, f)

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
            if save_figs:
                ax.cla()
                ax.pcolormesh(
                    xcm.cpu(), ycm.cpu(), rho.cpu(), shading="auto", cmap="jet"
                )
                ax.axis("equal")
                plt.show(block=False)
                plt.pause(0.1)
                fig.savefig(os.path.join(outAuxDir, "cfig.png"))

            print(
                "iter %d, t = [%.4e], dt = [%.4e], cpuTime = [%.2e]"
                % (iterFull, t, info.dt, (time.perf_counter() - tic))
            )
            # print(rho.max())
            tic = time.perf_counter()

            # if iterFull >= 100:
            #     return

            save_file(
                {
                    "u": u,
                    "xcm": xcm.clone().detach(),
                    "ycm": ycm.clone().detach(),
                    "t": torch.as_tensor(t),
                },
                os.path.join(outDir, f"solution_{iterFull:05d}.safetensors"),
                metadata={"info": json.dumps(info.toJson())},
            )

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
        figc.savefig(os.path.join(outAuxDir, "rho_t_%g.png" % (t)))

        iOut += 1
        if iOut >= len(tOuts):
            break
        pass


if __name__ == "__main__":
    Run()
