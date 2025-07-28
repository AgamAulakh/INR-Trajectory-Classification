import time
import torch
import random

N_SAMPLE_TOLERANCE = 10 # samples

def create_coordinate_grid(img, age, n_input):
    # if number of inputs is == 3, use 2d spatial dimension for all images in batch, select centre axial slice
    if n_input == 3:
        if len(img.shape) == 5:
            img = img[:, :, :, :, img.shape[-1] // 2]
        assert(len(img.shape) == 4)

    device = img.device
    mesh_vs = [age.flatten()] + [
        torch.arange(s, device=device) - s / 2 for s in img.shape[2:]
    ]
    coordinate_grid = (
        torch.stack(torch.meshgrid(mesh_vs, indexing="ij"), dim=-1)
        .unsqueeze(0)
        .reshape(list(img.shape) + [-1])
    )
    coordinate_grid[..., 1:] /= max(
        img.shape[2:]
    )  # normalize everything except age, which is already normalized
    return coordinate_grid

def create_sampling_mask(
    img,
    foreground_mask,
    sample_subj_voxels_ratio,
    sampling_foreground_ratio,
    sampling_context_seen
):
    device = img.device
 
    n = img.numel()  # total number of voxels in batch
    n_sample = int(n * sample_subj_voxels_ratio)
    background_mask = ~foreground_mask
    if sampling_context_seen is not None:
        # remove the context (coordinates) that we've already seen
        foreground_mask = foreground_mask & (~sampling_context_seen)
        background_mask = background_mask & (~sampling_context_seen)

    # count number of foreground pixels per image and over batch
    n_foreground_batch = torch.einsum("i...->i", foreground_mask)  # shape b
    n_background_batch = torch.einsum("i...->i", background_mask)  # shape b
    if sampling_foreground_ratio is not None:
        # NOTE: if (n * batch_ratio * fg_ratio) > n_foreground, sample all foreground voxels in batch
        n_sample_foreground = torch.minimum(
            torch.tensor(
                n_sample * sampling_foreground_ratio, dtype=int, device=device
            ).expand(img.shape[0]),
            n_foreground_batch,
        )
        n_sample_background = torch.minimum(
            torch.tensor(
                n_sample * (1 - sampling_foreground_ratio), dtype=int, device=device
            ).expand(img.shape[0]),
            n_background_batch,
        )
    else:
        n_sample_foreground = n_foreground_batch
        n_sample_background = n_background_batch

    sampling_mask = torch.zeros_like(
        foreground_mask, dtype=torch.bool
    )  # shape b, c, x, y, z

    # randomly shuffle foreground and background samples, select n < n_foreground or n_background
    for b in range(img.shape[0]):
        vf = (
            torch.randperm(n_foreground_batch[b], device=device)
            < n_sample_foreground[b]
        )  # shape n_foreground_batch
        sampling_mask[b, foreground_mask[b, ...]] = vf
        vb = (
            torch.randperm(n_background_batch[b], device=device)
            < n_sample_background[b]
        )  # shape n_background_batch
        sampling_mask[b, background_mask[b, ...]] = vb

    return sampling_mask


def sample_voxels(
    img,
    age=None,
    n_input=4,
    coordinate_grid=None,
    sample_subj_voxels_ratio=1.0,
    sampling_foreground_ratio=None,
):
    '''
    do not process whole image, but a percentage of the pixels

    sample_subj_voxels_ratio:  defines how many pixels are sampled per image for the training
    sampling_foreground_ratio: defines how many of the sampled pixels should be in the foreground (nonzero)
    '''
    raw = dict()

    # if number of inputs is == 3, use 2d spatial dimension for all images in batch, select centre axial slice
    if n_input == 3:
        if len(img.shape) == 5:
            img = img[:, :, :, :, img.shape[-1] // 2]
        assert(len(img.shape) == 4)

    foreground_mask = img >= -0.95
    if coordinate_grid is None:
        assert age is not None
        coordinate_grid = create_coordinate_grid(img, age, n_input)

    # sample groundtruth data
    sampling_mask = create_sampling_mask(
        img,
        foreground_mask,
        sample_subj_voxels_ratio,
        sampling_foreground_ratio,
        sampling_context_seen=None,
    )

    # flattening channels, each "channel" of image is actually an image intensity at a distinct age
    batch_pixel = img[sampling_mask].reshape(
        img.shape[0], -1, 1
    )  # shape b, samples, 1 output
    batch_coordinates = coordinate_grid[sampling_mask, :].reshape(
        img.shape[0], -1, coordinate_grid.shape[-1]
    )
    batch_pixel_foreground = foreground_mask[sampling_mask].reshape(img.shape[0], -1, 1)

    raw["batch_pixel_foreground"] = batch_pixel_foreground
    raw["sampling_mask"] = sampling_mask

    return batch_pixel, batch_coordinates, raw
