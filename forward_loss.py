import torch
from sample_voxels import sample_voxels

PEAK = 2  # normalized image dynamic range


def forward_loss(
    net,
    data,
    n_input,
    subj_idx,
    batch_ratio=None,
    presampled_batch_pixel=None,
    presampled_batch_coordinates=None,
    presampled_batch_raw=None,
    sampling_foreground_ratio=None,
    device="cpu",
):
    # sample pixels from image, compute forward pass, compute loss
    img = data["image"]
    age = data["age"]

    if (
        presampled_batch_coordinates is None
        and presampled_batch_coordinates is None
        and presampled_batch_raw is None
    ):
        assert batch_ratio is not None
        batch_pixel, batch_coordinates, sample_raw = sample_voxels(
            img=img,
            age=age,
            n_input=n_input,
            sample_subj_voxels_ratio=batch_ratio,
            sampling_foreground_ratio=sampling_foreground_ratio,
        )
    else:
        assert (
            presampled_batch_coordinates is not None
            and presampled_batch_coordinates is not None
            and presampled_batch_raw is not None
        )
        batch_pixel = presampled_batch_pixel
        batch_coordinates = presampled_batch_coordinates
        sample_raw = presampled_batch_raw

    # fwd pass
    pred_batch_pixel = net(batch_coordinates)

    # calc loss
    loss = (batch_pixel - pred_batch_pixel).pow(2)
    per_batch_loss = loss.mean()

    with torch.no_grad():
        # calc other metrics that wont be used for optimization
        loss_foreground = loss[sample_raw["batch_pixel_foreground"]].mean()
        loss_background = loss[~sample_raw["batch_pixel_foreground"]].mean()
        per_image_loss = loss.mean(dim=1).squeeze(-1)

        # calc psnr
        peak = torch.tensor(PEAK, dtype=int, device=device)
        psnr = (20 * torch.log10(peak) - 10 * torch.log10(per_batch_loss)).mean()

    losses = {
        "total": per_batch_loss,
        "foreground": loss_foreground,
        "background": loss_background,
        "mean_psnr": psnr,
        "total_by_image": per_image_loss,
    }
    return losses
