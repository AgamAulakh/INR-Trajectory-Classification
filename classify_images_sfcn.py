import argparse
import math
import os
import pathlib
import random
import time
import numpy as np
import pandas as pd
import torch

from construct_dataset import UKBBDataset
from models.SFCN import SFCNModel
from utils import get_prediction_accuracy

LR = 1e-4
WD = 1e-2

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Start testing on device:{device}")

    report_name = (
        f"classify_images_sfcn_b{args.batch_size}_e{args.n_epoch}"
    )
    # report_name = f"classify_inr_{args.preprocessed_data_file.split("/")[-1][:-3]}_ORIGINAL"
    if not os.path.exists(args.save_checkpoint_to_dir):
        os.makedirs(args.save_checkpoint_to_dir)

    datawrapper = UKBBDataset(
        device,
        "train",
        preprocessed_data_path=args.preprocessed_data_file,
        is_img_cropped=True,
        is_cache_enabled=True,
        is_data_padded=False if args.n_timepoints==4 else True,
    )

    train_loader = None
    test_loader = None
    val_loader = None

    _, train_loader = datawrapper.create_dataset_dataloader(
        args.batch_size, cache_rate=1.0
    )
    # if datawrapper.data_split.test_data_dicts is not None:
    #     _, test_loader = datawrapper.create_dataset_dataloader(args.batch_size, "test", cache_rate=1.0)
    if datawrapper.data_split.val_data_dicts is not None:
        _, val_loader = datawrapper.create_dataset_dataloader(
            args.batch_size, "val", cache_rate=0.0, use_cache_override=True,
        )

    # make classifier
    net = SFCNModel(args.n_timepoints)
    net.to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)

    start_epoch = 0
    best_epoch_mean_loss = 1000000.0
    best_epoch_mean_acc = 0.0
    epochs_no_loss_improvement = 0
    best_network_path = os.path.join(args.save_checkpoint_to_dir, report_name)
    best_network = {}
    best_network_acc = {}
    report_loss_over_time = []
    train_batch_count = math.floor(
        len(train_loader.dataset) / args.batch_size / args.n_grad_step
    )
    val_batch_count = math.ceil(len(val_loader.dataset) / args.batch_size)

    if args.load_checkpoint_from_file:
        best_network = torch.load(args.load_checkpoint_from_file, weights_only=False)
        net.load_state_dict(best_network["network_state_dict"])
        start_epoch = best_network["epoch"] + 1

    for epoch in range(start_epoch, args.n_epoch):
        epoch_stats = {
            "train_losses": torch.zeros([train_batch_count], device=device),
            "train_accs": torch.zeros([train_batch_count], device=device),
            "val_losses": torch.zeros([val_batch_count], device=device),
            "val_accs": torch.zeros([val_batch_count], device=device),
        }
        t_epoch_start = time.time()        
        # reset in case accumulating gradients
        opt.zero_grad()
        grad_iters = 0
        for iter, batch in enumerate(train_loader):
            if len(batch["image"]) != args.batch_size:
                continue

            logits = net(batch["image"])
            loss = torch.nn.functional.cross_entropy(logits, batch["fake_traj"])
            loss.backward()
            if (iter + 1) % args.n_grad_step == 0:
                opt.step()
                opt.zero_grad()

                pred_stats = get_prediction_accuracy(logits, batch["fake_traj"])
                epoch_stats["train_losses"][grad_iters] = loss
                epoch_stats["train_accs"][grad_iters] = pred_stats["accuracy"]
                grad_iters += 1
                if args.verbose:
                    print(
                        f"{epoch=} train {iter=} loss: {loss.item():10.8f} | "
                        f"acc: {pred_stats['accuracy']:.4f}"
                    )

        with torch.no_grad():
            for iter, batch in enumerate(val_loader):
                logits = net(batch["image"])
                loss = torch.nn.functional.cross_entropy(logits, batch["fake_traj"])
                pred_stats = get_prediction_accuracy(logits, batch["fake_traj"])

                print(
                    f"{epoch=} val {iter=} loss: {loss.item():10.8f} | "
                    f"tp: {pred_stats['tp']}, tn: {pred_stats['tn']}, fp: {pred_stats['fp']}, fn: {pred_stats['fn']} | "
                    f"acc: {pred_stats['accuracy']:.4f}"
                )
                epoch_stats["val_losses"][iter] = loss
                epoch_stats["val_accs"][iter] = pred_stats["accuracy"]

            epoch_mean_loss = epoch_stats["val_losses"].mean().item()
            epoch_mean_acc = epoch_stats["val_accs"].mean().item()
            dt_epoch = time.time() - t_epoch_start

            print(
                f"[{epoch=}] {dt_epoch=:.4f}] mean loss: {epoch_mean_loss:10.6f} mean acc: {epoch_mean_acc:10.6f}"
            )
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
                torch.save(checkpoint_network, best_network_path + f"_e{epoch}.pt")

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
                torch.save(best_network, best_network_path + "_best_loss.pt")
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
            for iter, batch in enumerate(test_loader):
                logits = net(batch["image"])
                probs = torch.nn.functional.softmax(logits, dim=1)
                pred_classes = torch.argmax(probs, dim=1)
                preds_correct += (pred_classes == batch["fake_traj"]).tolist()
                preds += (pred_classes).tolist()
                gt += (batch["fake_traj"]).tolist()
                subjs += batch["subject"]

            gt = np.array(gt)
            preds = np.array(preds)
            TP = np.sum((preds == 1) & (gt == 1))
            TN = np.sum((preds == 0) & (gt == 0))
            FP = np.sum((preds == 1) & (gt == 0))
            FN = np.sum((preds == 0) & (gt == 1))

            # test stats
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
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
        "--n-timepoints",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--n-epoch-early-stop",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--n-grad-step",
        type=int,
        default=10,
        help="number of gradient accumulation steps, effective batch size = batch size * n steps",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--preprocessed-data-file",
        type=str,
        default=os.path.join(
            pathlib.Path.home(),
            "Research/Projects/INR-Trajectory-Classification/data_splits/hdd_2x500subjs/traj_sim_1000subjs_t120v20v60_constSampleFalse.pt",
        ),
        help="Directory where data will be loaded from",
    )
    parser.add_argument(
        "--save-checkpoint-to-dir",
        type=str,
        default=os.path.join(
            pathlib.Path.home(),
            "Research/Projects/INR-Trajectory-Classification/checkpoint_traj_sim_sfcn/",
        ),
        help="Directory where checkpoints will be saved to",
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

    _ = main(args)
