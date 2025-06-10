from eulerCart.EulerCartDataset import (
    count_dataset_files,
    EulerCartTimeHistoryDataset,
    EulerCart_SNT_pre_process,
    EulerCart_SNT_F_MP,
)

from eulerCart.SmallNeigTransformer import SmallNeigTransformer

import torch, os
import time

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    torch.multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Already set, ignore


def train(
    dataset_loc,
    out_dir,
    bs=1,
    bs_val=4,
    lr=1e-3,
    n_epoch=4,
    device="cuda",
    seed=0,
    max_num_train_file=100000000,
    num_val=32,
    n_iter_val=100,
):
    if seed is not None:
        torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    num_dataset_file = count_dataset_files(dataset_loc)
    num_train = min(max_num_train_file, num_dataset_file - num_val)

    dataset = EulerCartTimeHistoryDataset(
        dir=dataset_loc,
        pre_processor=EulerCart_SNT_pre_process,
        # device="cpu", #move later
        device=device,
        min_i_file=0,
        max_i_file=num_train,
    )
    dataset_val = EulerCartTimeHistoryDataset(
        dir=dataset_loc,
        pre_processor=EulerCart_SNT_pre_process,
        # device="cpu", #move later
        device=device,
        min_i_file=num_dataset_file - num_val,
        max_i_file=num_dataset_file,
    )
    log_writer = SummaryWriter(out_dir)
    train_loader = DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=bs,
        persistent_workers=True,
        num_workers=min(os.cpu_count() // 2, bs),
    )
    val_loader = DataLoader(
        dataset=dataset_val,
        shuffle=False,
        batch_size=bs_val,
    )
    model = SmallNeigTransformer(
        dim_phy_in=4 + 1,
        dim_phy=4,
        dim=32,
        ge_dim=3,
        num_heads=1,
        ff_hidden_dim=64,
        num_encoder_layers=2,
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    loss_history = []
    g_step = 0
    tic = time.perf_counter()
    val_loss = float("nan")
    for i_epoch in range(n_epoch):
        model.train()
        epoch_loss = 0.0
        for i, (samples, ge_ins, labels) in enumerate(train_loader):
            samples = samples.to(device)
            ge_ins = ge_ins.to(device)
            labels = labels.to(device)
            g_step += 1
            optimizer.zero_grad()
            output = model(samples, ge_ins, EulerCart_SNT_F_MP)
            loss = torch.nn.functional.mse_loss(output, labels)
            loss.backward()
            loss_f = float(loss.cpu())
            epoch_loss += loss_f
            optimizer.step()

            loss_history.append((i_epoch, i, loss_f, epoch_loss))
            log_writer.add_scalar("Loss_Iter/train", loss_f, g_step)

            if i % n_iter_val == 0 or i == len(train_loader) - 1:
                model.eval()
                val_loss = 0.0
                for ii, (samples, ge_ins, labels) in enumerate(val_loader):
                    samples = samples.to(device)
                    ge_ins = ge_ins.to(device)
                    labels = labels.to(device)
                    output = model(samples, ge_ins, EulerCart_SNT_F_MP)
                    loss = torch.nn.functional.mse_loss(output, labels)
                    loss_f_eval = float(loss.cpu())
                    val_loss += loss_f_eval
                    print(f"Validating: {ii:6}/{len(val_loader)}")
                val_loss /= len(val_loader)
                log_writer.add_scalar("Loss_Iter/val", val_loss, g_step)
                model.train()

            if i % 1 == 0:
                timeC = time.perf_counter() - tic
                print(
                    f"{i_epoch+1:2}, {i+1:6}/{len(train_loader)}, loss: {loss_f:.6e}, val loss: {val_loss:.6e} time: {timeC:.3g}"
                )
                tic = time.perf_counter()

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"{i_epoch+1}, loss: {avg_epoch_loss:.6e}")
        loss_history.append((i_epoch, -1, 0.0, avg_epoch_loss))

        log_writer.add_scalar("Loss/train", avg_epoch_loss, i_epoch)
        log_writer.add_scalar("Loss/val", val_loss, i_epoch)

        torch.save(
            {
                "epoch": i_epoch,
                "model_state_dict": model.state_dict(),
                "opt_state_dict": optimizer.state_dict(),
                "loss": epoch_loss / len(train_loader),
                "history": loss_history,
            },
            os.path.join(out_dir, f"ckpt_{i_epoch+1:02}.pth"),
        )

    log_writer.close()


if __name__ == "__main__":
    train(
        "/home/harry/ssd1/data/eulerCart/box_1024-1",
        out_dir="out_0/train_0",
        bs=1,
        n_epoch=10,
        # max_num_file=12,
    )
