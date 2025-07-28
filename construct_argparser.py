import os
import argparse
import pathlib

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network",
        default="fullresidual",
        choices=["none", "fullresidual", "skip2residual"]
    )
    parser.add_argument(
        "--n-input",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--n-time-layers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--n-epoch",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--seed", 
        type=int,
        default=5,
    )
    parser.add_argument(
        "--fratio", 
        type=float,
        default=None,
    )
    parser.add_argument(
        "--activation",
        default="wire",
        choices=["wire", "sine", "relu","complexwire"]
    )
    parser.add_argument(
        "--time-activation",
        default="relu",
        choices=["wire", "sine", "relu","complexwire"]
    )
    parser.add_argument(
        "--wire-omega-0", 
        type=int,
        default=10,
    )
    parser.add_argument(
        "--wire-sigma-0", 
        type=int,
        default=10,
    )
    parser.add_argument(
        "--pos-encoding",
        default="none",
        choices=["none", "nerfoptimized", "gaussian"]
    )
    parser.add_argument(
        "--freq-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--n-freq",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--report", default=False, action="store_true",
    )
    parser.add_argument(
        "--verbose", default=False, action="store_true",
    )
    parser.add_argument(
        "--early-stopping-iters", type=int, default=-1,
    )
    parser.add_argument(
        "--simulate-batch-size", type=int, default=1,
    )
    parser.add_argument(
        "--save-checkpoint-to-dir",
        type=str,
        default=os.path.join(
            pathlib.Path.home(), "Research/Projects/INR-Trajectory-Classification/checkpoint/"
        ),
        help="Directory where checkpoints will be saved to",
    )
    parser.add_argument(
        "--load-checkpoint-from-file",
        type=str,
        default=False,
        help="torch file where checkpoints will be loaded from",
    )
    parser.add_argument(
        "--adapt-to-subj", default=False, action="store_true",
    )
    parser.add_argument(
        "--preprocessed-data-file",
        type=str,
        default=os.path.join(
            pathlib.Path.home(), "Research/Projects/INR-Trajectory-Classification/data_splits/traj_sim_alltrain_200subjs.pt"
        ),
        help="Directory where data will be loaded from",
    )
    parser.add_argument(
        "--use-interpol-extrapol",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--use-sep-spacetime",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--save-checkpoints-at-epoch",
        type=int,
        nargs='+',
    )
    parser.add_argument(
        "--scan",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--subj",
        type=int,
        default=-1,
    )
    return parser.parse_known_args()