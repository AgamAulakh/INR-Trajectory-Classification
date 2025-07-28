import argparse
import glob
import pathlib
import random
import time
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity as ssim
from monai.config import print_config
from monai.utils import misc
from monai.data import (
    ITKWriter,
    Dataset,
    CacheDataset,
    SmartCacheDataset,
    DataLoader,
)
from sample_voxels import (
    create_coordinate_grid,
    sample_voxels,
)
from construct_dataset import AbstractDataset, UKBBDataset
from inr.network import Network


def visualize_voxel_predictions_itk(raw, dir, filename, is_gt_needed=True, n_img=0):
    writer = ITKWriter()
    if is_gt_needed:
        writer.set_data_array(torch.squeeze(raw["batch_pixel"][n_img, ...].data), channel_dim=None)
        writer.write(os.path.join(dir, "gt_" + filename + ".nii.gz"))
    writer.set_data_array(torch.squeeze(raw["pred_batch_pixel"][:,n_img, ...].data), channel_dim=None)
    writer.write(os.path.join(dir, "pred_" + filename + ".nii.gz"))


def reconstruct_full_image(
    net,
    data,
    subj_idx,
    n_input,
    device="cpu",
    plane=None,
    mask_image=True,
):
    # sample pixels from image, compute forward pass, compute loss
    img = data["image"]
    mask = None
    if mask_image:
        mask = data["mask"]
        img = img * mask
    age = data["age"]
    raw = dict()

    # temp, use 3d spatial dimension for all images in batch, select centre axial slice
    is_2d_input = n_input == 3
    is_single_channel = img.shape[1] == 1

    sample_ratio = 0.005
    if is_2d_input or is_single_channel:
        sample_ratio = 0.025

    if is_2d_input:
        img = img[:, :, :, :, img.shape[-1] // 2]

    coordinates = create_coordinate_grid(
        img=img,
        age=age,
        n_input=n_input,
    )

    # TODO: OPTIMIZE
    # slowly reconstruct entire image using subset of sampled pixels of size total*0.02
    original_shape = coordinates.shape
    coordinates_flat = coordinates.reshape(original_shape[0], -1, original_shape[-1])
    subset_size = int(coordinates_flat.shape[1] * sample_ratio)
    pred_batch_pixel = torch.empty(
        (1, coordinates_flat.shape[1], 1), dtype=torch.float32
    )
    for start_i in range(0, coordinates_flat.shape[1], subset_size):
        end_i = min(start_i + subset_size, coordinates_flat.shape[1])
        subset_coordinates = coordinates_flat[:, start_i:end_i, :]
        pred_subset = net(subset_coordinates)
        pred_batch_pixel[:, start_i:end_i, :] = pred_subset

    # TODO: remove magic numbers in reshape
    raw.update(
        pred_batch_pixel=pred_batch_pixel.reshape(original_shape[:-1] + (1,)),
        batch_pixel=img[0],
        batch_coordinates=coordinates,
    )
    return raw


def reconstruct_subject_images(
    net,
    device,
    subj_data,
    subj_idx,
    data_split,
    n_input,
    recon_dir,
    is_interpol_extrapol_on=False,
):
    real_ages = data_split.estimate_real_age_from_norm(subj_data["age"])
    # reconstruct all ground truth ages at once (takes a long time)
    raw = reconstruct_full_image(
        net, subj_data, subj_idx, n_input, device=device, mask_image=False
    )
    n_img = 0
    for real_age, norm_age in zip(real_ages, subj_data["age"]):
        visualize_voxel_predictions_itk(
            raw,
            recon_dir,
            f"real{real_age.item():.4f}_norm{norm_age.item():.4f}",
            is_gt_needed=True,
            n_img=n_img,
        )
        n_img += 1

    if is_interpol_extrapol_on is False:
        return

    # inter- and extrapolation ages
    max_age = 80
    min_age = 50
    max_range = max_age - min_age + 20
    recon_ages = np.linspace(min_age - 10, max_age + 10, max_range + 1)
    norm_recon_ages = data_split.normalize_ages_over_age_list(recon_ages)

    # use dummy dataset to overwrite ages to, select first age and set channels=1:
    data_to_visualize = subj_data
    data_to_visualize["image"] = data_to_visualize["image"][:, 0, ...].unsqueeze(1)
    for age, norm_age in zip(recon_ages, norm_recon_ages):
        if age in real_ages:
            # already reconstructed
            continue
        # ignore the fact there are multiple ages and image channels in data_to_vis
        data_to_visualize["age"] = torch.tensor(
            [norm_age], device=device, dtype=torch.float32
        )
        raw = reconstruct_full_image(
            net, data_to_visualize, subj_idx, n_input, device=device, mask_image=False
        )
        # visualize predictions without GT map (doesnt exist)
        visualize_voxel_predictions_itk(
            raw, recon_dir, f"real{age:.4f}_norm{norm_age:.4f}", is_gt_needed=False
        )


def reconstruct_default_image(net, device, dataloader, n_input, dir, report_name):
    data_to_visualize = misc.first(dataloader)
    raw = reconstruct_full_image(
        net, data_to_visualize, n_input, device=device, mask_image=False
    )
    visualize_voxel_predictions_itk(raw, dir, report_name, is_gt_needed=True)


def compute_metrics(raw, device):
    gt = raw["batch_pixel"].cpu()
    pred = raw["pred_batch_pixel"][0].squeeze(-1).cpu()
    loss = (gt - pred).pow(2)
    per_batch_loss = loss.mean()

    # for img in range(gt.shape[0]):
    #     loss_img = (gt[img,:] - pred[img,:]).pow(2).mean()
    #     peak = torch.tensor(2, dtype=int, device="cpu")
    #     psnr = (20 * torch.log10(peak) - 10 * torch.log10(loss_img)).mean()
    #     print(img, psnr)

    results = {"loss": per_batch_loss}
    # peak is 2 because image dynamic range is from -1 to 1
    peak = torch.tensor(2, dtype=int, device="cpu")
    psnr = (20 * torch.log10(peak) - 10 * torch.log10(per_batch_loss)).mean()
    ssim_score = ssim(
        gt.numpy(), pred.numpy(), data_range=2, channel_axis=0
    )
    results["psnr_2"] = psnr
    results["ssim_2"] = ssim_score
    return results


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_input = 4
    datawrapper = UKBBDataset(
        device,
        "train",
        preprocessed_data_path=args.preprocessed_data_file,
        is_img_cropped=True,
        is_cache_enabled=False,
    )
    subj_data = datawrapper.data_split.train_data_dicts

    net = Network(
        device=device,
        hidden_size=512,
        input_size=n_input,
        num_layers=8,
        num_time_layers=5,
        activation="wire",
        time_activation="relu",
        wire_omega_0=10,
        wire_sigma_0=30,
        mlp_type="fullresidual",
        pos_encoding="none",
        num_frequencies=1.0,
        freq_scale=1,
        use_sep_spacetime=True,
    )
    net.to(device)

    assert os.path.isdir(args.load_inrs_from_dir)
    inr_files = glob.glob(os.path.join(args.load_inrs_from_dir, "*.pt"))
    inr_files.sort()
    reco_stats = []

    for inr_file in inr_files:
        t_subj_start = time.time()
        basename = os.path.basename(inr_file)
        for subj in subj_data:
            # silly - going through all data
            if (basename.startswith(f"train_{subj['subject']}_")):
                print(basename)
                with torch.no_grad():
                    net_ckpt = torch.load(inr_file, weights_only=False)
                    net.load_state_dict(net_ckpt["network_state_dict"])

                    cached_data = SmartCacheDataset(
                        data=[subj], transform=datawrapper.transforms, cache_rate=1.0
                    )
                    dataset = Dataset(
                        data=cached_data, transform=datawrapper.post_cache_transforms
                    )

                    temporal_range = np.linspace(50, 75, 26)
                    subj_real_ages = datawrapper.data_split.estimate_real_age_from_norm(
                        subj["age"]
                    )
                    subj_min_age = min(subj_real_ages)
                    subj_max_age = max(subj_real_ages)
                    interpolate_ages = [
                        t
                        for t in temporal_range
                        if t > subj_min_age
                        and t < subj_max_age
                        and t not in subj_real_ages
                    ]
                    extrapolate_ages = [
                        t
                        for t in temporal_range
                        if t < subj_min_age
                        or t > subj_max_age
                        and t not in subj_real_ages
                    ]

                    subj_stat = {
                        "subj": subj["subject"],
                    }

                    # training data stats
                    subj_data_transformed = datawrapper.get_subj_specific_data(
                        subj["subject_idx"], dataset
                    )
                    raw = reconstruct_full_image(
                        net,
                        subj_data_transformed,
                        [subj["subject_idx"]],
                        n_input,
                        device=device,
                        mask_image=False,
                    )
                    results = compute_metrics(raw, device=device)
                    subj_stat.update(
                        loss=results["loss"].item(),
                        psnr_2=results["psnr_2"].item(),
                        ssim_2=results["ssim_2"].item(),
                    )

                    # interpolation stats
                    if len(interpolate_ages) > 0:
                        subj_dir = "/".join(subj["image"][0].split("/")[:-1])
                        subj_base = os.path.basename(subj["image"][0][:-9])
                        interp_images = [
                            subj_dir + "/" + subj_base + str(int(t)) + ".nii.gz"
                            for t in interpolate_ages
                        ]
                        interp_data = {
                            "image": interp_images,
                            "age": datawrapper.data_split.normalize_ages_over_age_list(
                                interpolate_ages
                            ),
                            "subject": subj["subject"],
                            "subject_idx": subj["subject_idx"],
                            "fake_traj": subj["fake_traj"],
                        }
                        interp_data_transformed = datawrapper.transforms(interp_data)
                        interp_data_transformed = datawrapper.post_cache_transforms(
                            interp_data_transformed
                        )
                        interp_data_transformed["image"] = interp_data_transformed[
                            "image"
                        ].unsqueeze(0)
                        interp_data_transformed["age"] = interp_data_transformed[
                            "age"
                        ].unsqueeze(0)
                        raw = reconstruct_full_image(
                            net,
                            interp_data_transformed,
                            [subj["subject_idx"]],
                            n_input,
                            device=device,
                            mask_image=False,
                        )
                        results = compute_metrics(raw, device=device)
                        subj_stat.update(
                            interp_loss=results["loss"].item(),
                            interp_psnr_2=results["psnr_2"].item(),
                            interp_ssim_2=results["ssim_2"].item(),
                        )
                        # # reconstruction
                        # for n_img, (real_age, norm_age) in enumerate(zip(interpolate_ages, interp_data["age"])):
                        #     visualize_voxel_predictions_itk(
                        #         raw,
                        #         f"checkpoint/{subj["subject"]}",
                        #         f"real{real_age.item():.4f}_norm{norm_age.item():.4f}_interp",
                        #         is_gt_needed=True,
                        #         n_img=n_img,
                        #     )
                    else:
                        subj_stat.update(
                            interp_loss=None,
                            interp_psnr_1=None,
                            interp_ssim_1=None,
                            interp_psnr_2=None,
                            interp_ssim_2=None,
                        )

                    # extrapolation stats
                    if len(extrapolate_ages) > 0:
                        subj_dir = "/".join(subj["image"][0].split("/")[:-1])
                        subj_base = os.path.basename(subj["image"][0][:-9])
                        extrap_images = [
                            subj_dir + "/" + subj_base + str(int(t)) + ".nii.gz"
                            for t in extrapolate_ages
                        ]
                        extrap_data = {
                            "image": extrap_images,
                            "age": datawrapper.data_split.normalize_ages_over_age_list(
                                extrapolate_ages
                            ),
                            "subject": subj["subject"],
                            "subject_idx": subj["subject_idx"],
                            "fake_traj": subj["fake_traj"],
                        }
                        extrap_data_transformed = datawrapper.transforms(extrap_data)
                        extrap_data_transformed = datawrapper.post_cache_transforms(
                            extrap_data_transformed
                        )
                        extrap_data_transformed["image"] = extrap_data_transformed[
                            "image"
                        ].unsqueeze(0)
                        raw = reconstruct_full_image(
                            net,
                            extrap_data_transformed,
                            [subj["subject_idx"]],
                            n_input,
                            device=device,
                            mask_image=False,
                        )
                        results = compute_metrics(raw, device=device)
                        subj_stat.update(
                            extrap_loss=results["loss"].item(),
                            extrap_psnr_2=results["psnr_2"].item(),
                            extrap_ssim_2=results["ssim_2"].item(),
                        )
                        # # reconstruction
                        # for n_img, (real_age, norm_age) in enumerate(zip(extrapolate_ages, extrap_data["age"])):
                        #     visualize_voxel_predictions_itk(
                        #         raw,
                        #         f"checkpoint/{subj["subject"]}",
                        #         f"real{real_age.item():.4f}_norm{norm_age.item():.4f}_extrap",
                        #         is_gt_needed=True,
                        #         n_img=n_img,
                        #     )
                    else:
                        subj_stat.update(
                            extrap_loss=None,
                            extrap_psnr_1=None,
                            extrap_ssim_1=None,
                            extrap_psnr_2=None,
                            extrap_ssim_2=None,
                        )

                    subj_stat.update(
                        dt=time.time() - t_subj_start,
                        training_ages_norm=subj["age"],
                        training_ages=subj_real_ages,
                        interpol_ages=interpolate_ages,
                        extrapol_ages=extrapolate_ages,
                    )
                    reco_stats.append(subj_stat)
                    pd.DataFrame(reco_stats).to_csv(
                        os.path.join(
                            args.save_results_to_dir,
                            "inr_reconstruction_quality.csv",
                        )
                    )


def summarize(args):
    assert os.path.isdir(args.save_results_to_dir)
    results_file = os.path.join(
        args.save_results_to_dir,
        "inr_reconstruction_quality.csv",
    )
    assert os.path.isfile(results_file)

    df = pd.read_csv(results_file)

    phenotypes=["healthy", "adlike"]
    for phenotype in phenotypes:
        filtered_df = df[df["subj"].str.contains(phenotype, case=False, na=False)]

        stats={}
        stats["loss"] = filtered_df["loss"].mean()
        stats["psnr_2"] = filtered_df["psnr_2"].mean()
        stats["ssim_2"] = filtered_df["ssim_2"].mean()
        stats["interp_loss"] = filtered_df["interp_loss"].mean()
        stats["interp_psnr_2"] = filtered_df["interp_psnr_2"].mean()
        stats["interp_ssim_2"] = filtered_df["interp_ssim_2"].mean()
        stats["extrap_loss"] = filtered_df["extrap_loss"].mean()
        stats["extrap_psnr_2"] = filtered_df["extrap_psnr_2"].mean()
        stats["extrap_ssim_2"] = filtered_df["extrap_ssim_2"].mean()

        print(phenotype)
        print(stats)



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=5,
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
        "--save-results-to-dir",
        type=str,
        default=False,
        help="Directory where results will be saved to",
    )
    parser.add_argument(
        "--load-inrs-from-dir",
        type=str,
        default=False,
        help="Directory where subject INR checkpoints will be loaded from",
    )
    parser.add_argument(
        "--summarize-results",
        default=False,
        action="store_true",
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

    if args.save_results_to_dir is False:
        args.save_results_to_dir = args.load_inrs_from_dir

    if args.summarize_results:
        summarize(args)
    else:
        _ = main(args)
