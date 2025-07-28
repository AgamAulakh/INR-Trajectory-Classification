import copy
import math
import os
import random
import signal
import time

import numpy as np
import pandas as pd
import torch

from construct_argparser import get_args
from construct_dataset import UKBBDataset
from forward_loss import forward_loss
from inr.network import Network
from report import calc_sample_ratio, setup_report_name
from visualize_inr import reconstruct_default_image, reconstruct_subject_images


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Start training on device:{device}")

    datawrapper = UKBBDataset(
        device,
        "train",
        preprocessed_data_path=args.preprocessed_data_file,
        subj_to_fit=args.subj,
        scan_to_fit=args.scan,
        is_img_cropped=True,
        is_cache_enabled=(True),  # and args.subj<0),
        is_data_padded=False,
    )
    _, dataloader = datawrapper.create_dataset_dataloader(args.batch_size)
    subj_count = datawrapper.get_subject_count()
    batch_count = len(dataloader)
    net = Network(
        device=device,
        hidden_size=args.hidden_size,
        input_size=args.n_input,
        num_layers=args.n_layers,
        num_time_layers=args.n_time_layers,
        activation=args.activation,
        time_activation=args.time_activation,
        wire_omega_0=args.wire_omega_0,
        wire_sigma_0=args.wire_sigma_0,
        mlp_type=args.network,
        pos_encoding=args.pos_encoding,
        num_frequencies=args.n_freq,
        freq_scale=args.freq_scale,
    )
    net.to(device)
    batch_voxels_ratio, n_accumulation_steps = calc_sample_ratio(
        args, net
    )

    opt = torch.optim.AdamW(params=net.get_global_parameters(), lr=args.lr)

    subj_header = ""
    if args.subj > -1:
        subj_header = datawrapper.get_subj_name(args.subj)
    report_name = setup_report_name(args, "train", subj_header, batch_voxels_ratio)

    # set up best network to save, include data config
    best_epoch_mean_loss = 100000.0  # random value
    best_network_path = os.path.join(args.save_checkpoint_to_dir, report_name)
    best_network = {}

    # incase we have to simulate different batch sizes (for irregularly sampled data with inconsistent sizes)
    # Note: n_accumulation steps is the number of steps taken for one batch (accumulating sampled voxels)
    #       n_steps_over_batches is the number batches the gradient is accumulated over
    n_steps_over_batches = 1
    if args.simulate_batch_size > 1:
        n_steps_over_batches = args.simulate_batch_size // args.batch_size
    assert n_steps_over_batches >= 1, "if simulating larger batches, simulated_batch_size must be larger than batch_size used with dataloader"

    # set up training loop
    start_epoch = 0
    report_loss_over_time = []
    sigint_count = 0
    iters_no_loss_improvement = 0

    if args.load_checkpoint_from_file:
        best_network = torch.load(args.load_checkpoint_from_file, weights_only=False)
        net.load_state_dict(best_network["network_state_dict"])
        opt.load_state_dict(best_network["optimizer_state_dict"])
        start_epoch = best_network["epoch"] + 1
        if args.adapt_to_subj:
            # continue training entire model for subject adaptation
            # bad idea, but overwrite n_epochs
            args.n_epoch += start_epoch

    for epoch in range(start_epoch, args.n_epoch):
        net.train()
        epoch_losses = {
            "total": torch.zeros([batch_count // n_steps_over_batches], device="cpu"),
            "foreground": torch.zeros([batch_count // n_steps_over_batches], device="cpu"),
            "background": torch.zeros([batch_count // n_steps_over_batches], device="cpu"),
            "mean_psnr": torch.zeros([batch_count // n_steps_over_batches], device="cpu"),
        }
        t_epoch_start = time.time()

        # reset batch-wise accumulation:
        step_count = 0
        step_over_batch = 0
        opt.zero_grad()

        for batch in dataloader:
            if len(batch["image"]) != args.batch_size or step_count >= (batch_count // n_steps_over_batches):
                continue
            step_over_batch += 1

            for step in range(n_accumulation_steps):
                # scale the loss to the mean of the accumulated batch size
                losses = forward_loss(
                    net,
                    batch,
                    n_input=args.n_input,
                    subj_idx=list(range(args.batch_size)),
                    batch_ratio=batch_voxels_ratio,
                    sampling_foreground_ratio=args.fratio,
                    device=device,
                )
                loss = losses["total"] / float(n_accumulation_steps * n_steps_over_batches)

                # save this iter for epoch report
                epoch_losses["total"][step_count] += loss.item()
                epoch_losses["foreground"][step_count] += losses["foreground"].item() / float(
                    n_accumulation_steps * n_steps_over_batches
                )
                epoch_losses["background"][step_count] += losses["background"].item() / float(
                    n_accumulation_steps * n_steps_over_batches
                )
                epoch_losses["mean_psnr"][step_count] += losses["mean_psnr"].item() / float(
                    n_accumulation_steps * n_steps_over_batches
                )

                loss.backward()
                if step + 1 == n_accumulation_steps and step_over_batch == n_steps_over_batches:
                    opt.step()
                    opt.zero_grad()
                    step_over_batch = 0
                    step_count += 1

        # end of epoch report
        net.eval()
        with torch.no_grad():
            if args.report:
                report_loss = {
                    key: value.mean().item() for key, value in epoch_losses.items()
                }
                report_loss["epoch"] = epoch
                report_loss["dt_epoch"] = time.time() - t_epoch_start
                report_loss_over_time.append(report_loss)

                # save every epoch, not at end (remove this entirely)
                pd.DataFrame(report_loss_over_time).to_csv(
                    os.path.join(
                        args.save_checkpoint_to_dir,
                        report_name + ".csv",
                    )
                )
            if args.verbose:
                epoch_loss = epoch_losses["total"].mean().item()
                epoch_loss_fg = epoch_losses["foreground"].mean().item()
                epoch_loss_bg = epoch_losses["background"].mean().item()
                epoch_psnr = epoch_losses["mean_psnr"].mean().item()
                dt_epoch = report_loss["dt_epoch"]
                print(
                    f"{epoch=} {dt_epoch=:.4f} Epoch loss: {epoch_loss:10.8f} loss_fg: {epoch_loss_fg:10.8f} loss_bg: {epoch_loss_bg:10.8f} mean_psnr: {epoch_psnr:10.8f}"
                )

            # save the best model
            if epoch_losses["total"].mean().item() < best_epoch_mean_loss:
                best_epoch_mean_loss = epoch_losses["total"].mean().item()
                if args.report:
                    best_network.update(
                        epoch=epoch,
                        network_state_dict=net.state_dict(),
                        optimizer_state_dict=opt.state_dict(),
                    )
                    torch.save(best_network, best_network_path + ".pt")
            else:
                iters_no_loss_improvement += 1

            # save some checkpoints too
            if (
                args.save_checkpoints_at_epoch is not None
                and epoch + 1 in args.save_checkpoints_at_epoch
            ):
                checkpoint_network = {}
                checkpoint_network.update(
                    epoch=epoch,
                    network_state_dict=net.state_dict(),
                    optimizer_state_dict=opt.state_dict(),
                )
                torch.save(checkpoint_network, best_network_path + f"_e{epoch}.pt")

            # early stopping
            if (
                args.early_stopping_iters > 0
                and iters_no_loss_improvement >= args.early_stopping_iters
            ):
                print(
                    f"\nEarly stopping: no improvement for {iters_no_loss_improvement} training iterations"
                )
                break

    # end of training
    # save the report if it is needed
    if args.report:
        pd.DataFrame(report_loss_over_time).to_csv(
            os.path.join(
                args.save_checkpoint_to_dir,
                report_name + ".csv",
            )
        )
        torch.save(best_network, best_network_path + ".pt")

    print(f"Best epoch loss (MSE):{best_epoch_mean_loss:10.8f}")
    return best_epoch_mean_loss


if __name__ == "__main__":
    args, unknown = get_args()
    print(f"accepted arguments: {args}")
    if unknown:
        print(f"ERROR: ignoring {unknown}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    _ = main(args)
