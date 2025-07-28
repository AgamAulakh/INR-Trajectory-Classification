import argparse
import glob
import multiprocessing as mp
import math
import os
import pathlib
import time
import numpy as np
import random
import pandas as pd
import torch

import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns

from inr.network import Network
from models.inr2vec import Encoder
from construct_dataset import AbstractDataset, UKBBDataset
from construct_inr_dataset import construct_inr_dataloader
from report import setup_report_name, calc_sample_ratio
from utils import get_prediction_accuracy

LR = 1e-4
WD = 1e-2

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Start testing on device:{device}")

    report_name = f"classify_inr_e{args.n_epoch}_b{args.batch_size}_hd{args.hidden_dims[0]}"
    for d in args.hidden_dims[1:]:
        report_name+=f"-{d}"
    report_name+=f"_ed{args.embed_dim}"
    report_name+=f"_useSpace{args.use_space_layers}"
    report_name+=f"_useTime{args.use_time_layers}"
    report_name+=f"_useCombined{args.use_combined_layers}"
    report_name+=f"_shuffleCols{args.shuffle_cols}"

    if not os.path.exists(args.save_checkpoint_to_dir):
        os.makedirs(args.save_checkpoint_to_dir)

    datawrapper = UKBBDataset(
        device,
        "train",
        preprocessed_data_path=args.preprocessed_data_file,
        is_img_cropped=True,
        is_cache_enabled=False,
    )

    train_loader = construct_inr_dataloader(
        datawrapper.data_split.train_data_dicts,
        args.load_inrs_from_dir,
        args.net_config_file,
        args.use_space_layers,
        args.use_time_layers,
        args.use_combined_layers,
        batch_size=args.batch_size,
        shuffle_inrs=True,
        shuffle_cols=args.shuffle_cols,
    )
    test_loader = None
    val_loader = None
    if datawrapper.data_split.test_data_dicts is not None:
        test_loader = construct_inr_dataloader(
            datawrapper.data_split.test_data_dicts,
            args.load_inrs_from_dir,
            args.net_config_file,
            args.use_space_layers,
            args.use_time_layers,
            args.use_combined_layers,
            batch_size=args.batch_size,
        )

    if datawrapper.data_split.val_data_dicts is not None:
        val_loader = construct_inr_dataloader(
            datawrapper.data_split.val_data_dicts,
            args.load_inrs_from_dir,
            args.net_config_file,
            args.use_space_layers,
            args.use_time_layers,
            args.use_combined_layers,
            batch_size=args.batch_size,
        )

    # make classifier
    net = Encoder(input_dim=512, hidden_dims=args.hidden_dims,embed_dim=args.embed_dim)
    net.to(device)

    num_steps = args.n_epoch * len(train_loader)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=num_steps)

    start_epoch = 0
    best_epoch_mean_loss = 1000000.0
    best_epoch_mean_acc = 0.0
    epochs_no_loss_improvement = 0
    best_network_path = os.path.join(args.save_checkpoint_to_dir, report_name)
    best_network = {}
    best_network_acc = {}
    report_loss_over_time = []
    train_batch_count = math.floor(
        len(train_loader.dataset) / args.batch_size
    )
    val_batch_count = math.ceil(
        len(val_loader.dataset) / args.batch_size
    )

    if args.load_checkpoint_from_file:
        best_network = torch.load(args.load_checkpoint_from_file, weights_only=False)
        net.load_state_dict(best_network["network_state_dict"])
        start_epoch = best_network["epoch"] + 1

    # TODO:
    # if train and val data loaders are not none, run this, else assert test loader is not none and run that
    for epoch in range(start_epoch, args.n_epoch):
        epoch_stats = {
            "train_losses": torch.zeros([train_batch_count], device=device),
            "train_accs": torch.zeros([train_batch_count], device=device),
            "val_losses": torch.zeros([val_batch_count], device=device),
            "val_accs": torch.zeros([val_batch_count], device=device),
        }
        t_epoch_start = time.time()

        for iter, (inrs, _, _, labels) in enumerate(train_loader):
            if len(labels) != args.batch_size:
                continue

            logits, _ = net(inrs)
            loss = torch.nn.functional.cross_entropy(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

            pred_stats = get_prediction_accuracy(logits, labels)

            epoch_stats["train_losses"][iter] = loss
            epoch_stats["train_accs"][iter] = pred_stats["accuracy"]

            if args.verbose:
                print(
                    f"{epoch=} train {iter=} loss: {loss.item():10.8f} | "\
                    f"acc: {pred_stats["accuracy"]:.4f}"
                )
        with torch.no_grad():
            for iter, (inrs, _, _, labels) in enumerate(val_loader):
                logits,_= net(inrs)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                pred_stats = get_prediction_accuracy(logits, labels)

                if args.verbose:
                    print(
                        f"{epoch=} val {iter=} loss: {loss.item():10.8f} | "\
                        f"tp: {pred_stats["tp"]}, tn: {pred_stats["tn"]}, fp: {pred_stats["fp"]}, fn: {pred_stats["fn"]} | "\
                        f"acc: {pred_stats["accuracy"]:.4f}"
                    )
                epoch_stats["val_losses"][iter] = loss
                epoch_stats["val_accs"][iter] = pred_stats["accuracy"]

            epoch_mean_loss = epoch_stats["val_losses"].mean().item()
            epoch_mean_acc = epoch_stats["val_accs"].mean().item()
            dt_epoch = time.time() - t_epoch_start

            print(f"[{epoch=}] {dt_epoch=:.4f} | mean loss: {epoch_mean_loss:10.6f} mean acc: {epoch_mean_acc:10.6f}")
            report_loss = {
                "train_loss": epoch_stats["train_losses"].mean().item(),
                "train_acc": epoch_stats["train_accs"].mean().item(),
                "val_loss": epoch_mean_loss,
                "val_acc": epoch_mean_acc,
                "epoch": epoch,
                "dt_epoch": dt_epoch,
            }
            report_loss_over_time.append(report_loss)

            # save every epoch, not at end (remove this entirely)
            pd.DataFrame(report_loss_over_time).to_csv(
                os.path.join(
                    args.save_checkpoint_to_dir,
                    report_name + ".csv",
                )
            )

            if (epoch+1) % 50 == 0 and epoch != 0:
                checkpoint_network = {}
                checkpoint_network.update(
                    epoch=epoch,
                    network_state_dict=net.state_dict(),
                    optimizer_state_dict=opt.state_dict(),
                )
                torch.save(checkpoint_network, best_network_path+f"_e{epoch}.pt")

            if epoch_mean_acc >= best_epoch_mean_acc:
                best_epoch_mean_acc = epoch_mean_acc
                epochs_no_loss_improvement = 0
                best_network_acc.update(
                    epoch=epoch,
                    network_state_dict=net.state_dict(),
                    optimizer_state_dict=opt.state_dict(),
                )
                torch.save(best_network_acc, best_network_path+"_best_acc.pt")
            if epoch_mean_loss <= best_epoch_mean_loss:
                best_epoch_mean_loss = epoch_mean_loss
                epochs_no_loss_improvement = 0
                best_network.update(
                    epoch=epoch,
                    network_state_dict=net.state_dict(),
                    optimizer_state_dict=opt.state_dict(),
                )
                torch.save(best_network, best_network_path+"_best_loss.pt")
            else:
                epochs_no_loss_improvement += 1

                if (
                    epochs_no_loss_improvement >= args.n_epoch_early_stop
                    and args.n_epoch_early_stop > 0
                    and epoch >= 99
                ):
                    print(
                        f"\nEarly stopping: no improvement for {epochs_no_loss_improvement} training iterations"
                    )
                    break

    net.eval()
    with torch.no_grad():
        if test_loader is not None:
            net.load_state_dict(best_network["network_state_dict"])
            best_epoch = best_network["epoch"]
            preds_correct = []
            preds = []
            gt = []
            subjs = []
            for iter, (inrs, subj, _, labels) in enumerate(test_loader):
                logits,_ = net(inrs)
                probs = torch.nn.functional.softmax(logits, dim=1)
                pred_classes = torch.argmax(probs, dim=1)
                preds_correct += (pred_classes == labels).tolist()
                preds += (pred_classes).tolist()
                gt += (labels).tolist()
                subjs += subj

            gt = np.array(gt)
            preds = np.array(preds)
            TP = np.sum((preds == 1) & (gt == 1))
            TN = np.sum((preds == 0) & (gt == 0))
            FP = np.sum((preds == 1) & (gt == 0))
            FN = np.sum((preds == 0) & (gt == 1))

            # test stats
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            print(TP, TN, FP, FN)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (TP + TN) / (TP + TN + FP + FN)

            print(f"Accuracy: {accuracy:.5f}")
            print(f"Precision: {precision:.5f}")
            print(f"Recall: {recall:.5f}")
            print(f"F1 Score: {f1:.5f}")

            test_acc = sum(preds_correct) / len(preds_correct)
            print(f"Net from e{best_epoch} test accuracy: {test_acc:.5f}%")
            print("Correct:")
            print([s for s, p in zip(subjs, preds_correct) if p])
            print("Incorrect:")
            incorrect = sorted([s for s, p in zip(subjs, preds_correct) if not p])
            print(incorrect)

            # pca_train = visualize_latent_space(net, train_loader, "train")
            # visualize_latent_space(net, test_loader, "test", pca=pca_train)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--n-epoch",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--n-epoch-early-stop",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
    )
    parser.add_argument(
        '--hidden-dims',
        type=int,
        nargs='+',
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--use-space-layers", default=False, action="store_true",
    )
    parser.add_argument(
        "--use-time-layers", default=False, action="store_true",
    )
    parser.add_argument(
        "--use-combined-layers", default=False, action="store_true",
    )
    parser.add_argument(
        "--shuffle-cols", default=False, action="store_true",
    )
    parser.add_argument(
        "--verbose", default=False, action="store_true",
    )
    parser.add_argument(
        "--net-config-file",
        type=str,
        default=os.path.join(
            pathlib.Path.home(), "Research/Projects/INR-Trajectory-Classification/net_config.yaml"
        ),
        help="network config file created before training",
    )
    parser.add_argument(
        "--preprocessed-data-file",
        type=str,
        default=os.path.join(
            pathlib.Path.home(),
            "Research/Projects/INR-Trajectory-Classification/data_splits/traj_sim_200subjs_t60v10t30.pt",
        ),
        help="Directory where data will be loaded from",
    )
    parser.add_argument(
        "--save-checkpoint-to-dir",
        type=str,
        default=False,
        help="Directory where checkpoints will be saved to",
    )
    parser.add_argument(
        "--load-inrs-from-dir",
        type=str,
        default=False,
        help="Directory where subject INR checkpoints will be loaded from",
    )
    parser.add_argument(
        "--load-checkpoint-from-file",
        type=str,
        default=False,
        help="torch file where checkpoints will be loaded from",
    )
    return parser.parse_known_args()


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

    if args.save_checkpoint_to_dir is False:
        args.save_checkpoint_to_dir = args.load_inrs_from_dir

    _ = main(args)
