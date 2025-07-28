import math
import os
import numpy as np
import yaml

def setup_report_name(args, data_split, subj_header, vratio):
    if not os.path.exists(args.save_checkpoint_to_dir):
        os.makedirs(args.save_checkpoint_to_dir)

    # saving the config to yaml
    net_config = {
        "hidden_size": args.hidden_size,
        "input_size": args.n_input,
        "num_layers": args.n_layers,
        "num_time_layers": args.n_time_layers,
        "num_subjects": args.batch_size,
        "activation": args.activation,
        "time_activation": args.time_activation,
        "wire_omega_0": args.wire_omega_0,
        "wire_sigma_0": args.wire_sigma_0,
        "mlp_type": args.network,
        "pos_encoding": args.pos_encoding,
        "num_frequencies": args.n_freq,
        "freq_scale": args.freq_scale,
        "use_sep_spacetime": args.use_sep_spacetime,
    }
    with open("net_config.yaml", "w") as f:
        yaml.dump(net_config, f)

    assert data_split in ("train", "test")
    included_data = f"{data_split}_"

    assert args.n_input == 3 or args.n_input == 4, (
        f"cannot use {args.n_input=}, must be 3 or 4 (2d+t or 3d+t)"
    )
    assert args.fratio is not None
    if args.subj > -1:
        assert args.batch_size <= 3 and args.batch_size > 0, (
            f"cannot use {args.subj=}, {args.batch_size=} (if using one subj, batch size must be # scans of subj)"
        )
        assert args.scan < 0, (
            f"cannot use {args.subj=}, {args.scan=} (only use one at a time for now)"
        )
        if args.adapt_to_subj:
            included_data += f"{subj_header}_adapt_"

    if args.scan > -1:
        assert args.batch_size == 1, (
            f"cannot use {args.scan=}, {args.batch_size=} (if using one scan, batch size must be 1)"
        )
        # redundant check:
        assert args.subj < 0, (
            f"cannot use {args.subj=}, {args.scan=} (only use one at a time for now)"
        )
        included_data = f"scan{args.scan}_"

    assert args.n_freq >= 0 and args.freq_scale >= 0, (
        f"cannot use {args.n_freq=}, {args.freq_scale=} (only use non-negative frequencies for positional encoding)"
    )

    report_name = (
        included_data + f"trajsim_{args.network}_"
        f"{args.n_input - 1}d_"
        f"nl{args.n_layers}_"
        f"ntimel{args.n_time_layers}_"
        f"hs{args.hidden_size}_"
        f"{args.activation}_"
        f"ta{args.time_activation}_"
        f"wo{args.wire_omega_0}_"
        f"ws{args.wire_sigma_0}_"
        f"posenc{args.pos_encoding}_"
        f"nfreq{args.n_freq}_"
        f"freqscale{args.freq_scale}_"
        f"e{args.n_epoch}_"
        f"b{args.batch_size}_"
        f"fratio{args.fratio}_"
        f"vratio{vratio}_"
        f"lr{args.lr}"
    )
    if args.use_sep_spacetime:
        report_name += "_use_sep_spacetime"

    # reassign n_freq
    if args.pos_encoding in ("nerfoptimized", "nerf"):
        num_frequencies = (
            args.n_freq,
            args.n_freq,
            args.n_freq,
            args.n_freq,
        )
        args.n_freq = num_frequencies
    else:
        args.n_freq = (args.n_freq,)

    return report_name


def calc_sample_ratio(args, net):
    # TODO: this may still underutilize the gpu
    if args.n_input == 4:
        if args.subj > -1:
            target_batch_voxels_ratio = 0.01
            batch_voxels_ratio = 0.01
        else:
            target_batch_voxels_ratio = 0.0012
            batch_voxels_ratio = 0.0012
        assert batch_voxels_ratio > 0
    else:
        if args.subj > -1:
            target_batch_voxels_ratio = 1.0
            batch_voxels_ratio = 1.0
        else:
            target_batch_voxels_ratio = 0.02
            batch_voxels_ratio = 0.02

    assert (
        (args.fratio is None and batch_voxels_ratio == 1.0)
        or (args.fratio is not None and batch_voxels_ratio < 1.0)
    )

    accumulation_steps = math.ceil(target_batch_voxels_ratio / batch_voxels_ratio)
    print(
        f"{batch_voxels_ratio=:10.5f}, "
        f"{target_batch_voxels_ratio=}, "
        f"{accumulation_steps=}"
    )

    return batch_voxels_ratio, accumulation_steps
