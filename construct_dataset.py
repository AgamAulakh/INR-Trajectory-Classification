import glob
import os
import abc
from math import pi
from pathlib import Path
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import pandas as pd

from monai.config import print_config
from monai.utils import misc
from monai.data import (
    Dataset,
    CacheDataset,
    SmartCacheDataset,
    DataLoader,
    ITKWriter,
)
from monai.transforms import (
    Compose,
    LoadImaged,
    MapTransform,
    Lambdad,
    ScaleIntensityRangePercentilesd,
    CenterSpatialCropd,
    ToTensord,
    Identityd,
    EnsureTyped,
)


class DataSplit:
    def __init__(self, age_1p, age_99p, is_age_clipped):
        self.train_data_dicts = None
        self.val_data_dicts = None
        self.test_data_dicts = None
        self.age_1p = age_1p
        self.age_99p = age_99p
        self.is_age_clipped = is_age_clipped

    def save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path):
        # dont need this to be obj specific
        return torch.load(path, weights_only=False)

    def normalize_ages_over_age_list(self, age_list: list[int]) -> list[int]:
        # using previously-found age percentiles:
        norm_age_list = (age_list - self.age_1p) / (self.age_99p - self.age_1p)
        if self.is_age_clipped:
            norm_age_list = np.clip(norm_age_list, 0, 1).tolist()
        return norm_age_list

    def normalize_ages_over_all_data(self) -> None:
        assert self.age_1p != 0 and self.age_99p != 0
        if self.train_data_dicts is not None:
            for subj_data in self.train_data_dicts:
                subj_data["age"] = (subj_data["age"] - self.age_1p) / (
                    self.age_99p - self.age_1p
                )
                if self.is_age_clipped:
                    subj_data["age"] = np.clip(subj_data["age"], 0, 1).tolist()
        if self.val_data_dicts is not None:
            for subj_data in self.val_data_dicts:
                subj_data["age"] = (subj_data["age"] - self.age_1p) / (
                    self.age_99p - self.age_1p
                )
                if self.is_age_clipped:
                    subj_data["age"] = np.clip(subj_data["age"], 0, 1).tolist()
        if self.test_data_dicts is not None:
            for subj_data in self.test_data_dicts:
                subj_data["age"] = (subj_data["age"] - self.age_1p) / (
                    self.age_99p - self.age_1p
                )
                if self.is_age_clipped:
                    subj_data["age"] = np.clip(subj_data["age"], 0, 1).tolist()

    def estimate_real_age_from_norm(self, age):
        # NOTE: this is only an estimate! might be wrong if normalized ages are clipped
        assert self.age_1p != 0 and self.age_99p != 0
        return age * (self.age_99p - self.age_1p) + self.age_1p


class AbstractDataset:
    def __init__(self, device, split="train", is_cache_enabled=False):
        assert split in ("train", "val", "test")

        self.device = device
        self.data_split = DataSplit(age_1p=0, age_99p=0, is_age_clipped=False)
        self.split_to_use = split
        self.img_size = [197, 223, 189]
        self.subject_count = 0
        self.max_imgs_per_subject = 4
        self.transforms = None
        self.post_cache_transforms = None
        self.is_cache_enabled = is_cache_enabled

    @abc.abstractmethod
    def create_transforms(self, is_img_cropped):
        raise NotImplementedError

    def get_subject_count(self):
        return self.subject_count

    def get_max_imgs_per_subject(self):
        return self.max_imgs_per_subject

    def get_subj_specific_data(self, subj_idx, dataset=None):
        if dataset is None:
            dataset = getattr(self.data_split, f"{self.split_to_use}_data_dicts")

        assert dataset is not None
        if isinstance(dataset, list):
            subj_samples = [
                i for i, d in enumerate(dataset) if d["subject_idx"] == subj_idx
            ]
        else:
            # assume it's a monai dataset
            subj_samples = [
                i for i, d in enumerate(dataset.data) if d["subject_idx"] == subj_idx
            ]
        if not subj_samples:
            raise ValueError(f"Subject IDX {subj_idx} not found in dataset.")
        subj_data = [dataset[i] for i in subj_samples][0]
        if not isinstance(dataset, list):
            # if it's a monai dataset
            subj_data["image"] = subj_data["image"].unsqueeze(0)  # add batch size = 1
        return subj_data

    def get_subj_name(self, subj_idx, dataset=None):
        subj_data = self.get_subj_specific_data(subj_idx, dataset)
        return subj_data["subject"]

    @staticmethod
    def get_ages_from_data_dict(data_dicts):
        return [age for subj in data_dicts for age in subj["age"]]


class UKBBDataset(AbstractDataset):
    def __init__(
        self,
        device,
        split="train",
        preprocessed_data_path=None,
        subj_to_fit=-1,
        scan_to_fit=-1,
        is_img_cropped=False,
        is_cache_enabled=True,
        is_data_padded=False,
    ):
        # TODO!!!!! refactor use of split_to_use

        super().__init__(device, split, is_cache_enabled)
        if is_img_cropped:
            self.img_size = [147, 183, 169]

        assert os.path.isfile(preprocessed_data_path)
        self.data_split = DataSplit.load(path=preprocessed_data_path)
        # limit datasplits to only have one subj if required
        if subj_to_fit > -1:
            filtered_data = self.get_subj_specific_data(subj_to_fit)
            setattr(self.data_split, f"{self.split_to_use}_data_dicts", [filtered_data])

        assert (
            self.data_split.train_data_dicts is not None
            or self.data_split.test_data_dicts is not None
        )
        if self.data_split.train_data_dicts is not None:
            self.subject_count = len(self.data_split.train_data_dicts)
        else:
            self.subject_count = len(self.data_split.test_data_dicts)

        self.create_transforms(is_img_cropped, is_data_padded)

    def create_transforms(self, is_img_cropped, is_data_padded):
        # define transform/load images
        if self.is_cache_enabled:
            # store cache on cpu, use post_cache_transforms to move to gpus
            device = "cpu"
        else:
            device = self.device
        self.transforms = Compose(
            [
                LoadImaged(
                    keys=["image"], ensure_channel_first=True, reader="ITKReader"
                ),
                ScaleIntensityRangePercentilesd(
                    keys="image", lower=1, upper=99, b_min=-1, b_max=1, clip=True
                ),
                CenterSpatialCropd(keys="image", roi_size=[147, 183, 169], lazy=True) if is_img_cropped else Identityd(keys=["image"]),
                ToTensord(keys=["image"], track_meta=False, device=device),
                Lambdad(
                    keys=["age"],
                    func=lambda x: torch.tensor(x, dtype=torch.float32, device=device),
                ),
                Lambdad(
                    keys=["subject_idx"],
                    func=lambda x: torch.tensor(x, dtype=torch.float32, device=device),
                ),
                Lambdad(
                    keys=["fake_traj"],
                    func=lambda x: torch.tensor(x, dtype=torch.long, device=device),
                ),
                PadSequenceToN(keys=["image"],n=5) if is_data_padded else Identityd(keys=["image"]),
                PadSequenceToN(keys=["age"], n=5) if is_data_padded else Identityd(keys=["age"]),
            ]
        )
        self.post_cache_transforms = Compose(
            [
                EnsureTyped(keys=["image"], device=self.device),
                EnsureTyped(keys=["age"], device=self.device),
                EnsureTyped(keys=["subject"], device=self.device),
                EnsureTyped(keys=["subject_idx"], device=self.device),
                EnsureTyped(keys=["fake_traj"], device=self.device),
            ]
        )

    def create_dataset_dataloader(
        self, batch_size, split_to_use=None, cache_rate=1.0, use_cache_override=False
    ):
        if split_to_use is None:
            data_to_load = getattr(self.data_split, f"{self.split_to_use}_data_dicts")
        else:
            assert split_to_use in ("train", "val", "test")
            data_to_load = getattr(self.data_split, f"{split_to_use}_data_dicts")

        # create dataset and dataloader
        dataset = None
        if self.is_cache_enabled and not use_cache_override:
            cached_data = SmartCacheDataset(
                data=data_to_load, transform=self.transforms, cache_rate=cache_rate
            )
            dataset = Dataset(data=cached_data, transform=self.post_cache_transforms)
        elif use_cache_override:
            # use basic data set but the correct transforms have not been setup for it
            cached_data = Dataset(data=data_to_load, transform=self.transforms)
            dataset = Dataset(data=cached_data, transform=self.post_cache_transforms)
        else:
            dataset = Dataset(
                data=data_to_load,
                transform=self.transforms,
            )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        return dataset, dataloader


class PadSequenceToN(MapTransform):
    # NOTE: must use this with tensors or lists of tensors, not file paths
    def __init__(self, keys, n=5, mode="zeros"):
        super().__init__(keys)
        self.n = n
        self.mode = mode

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            seq = d[k]

            if isinstance(seq, list):
                pad_len = self.n - len(seq)
                if pad_len > 0:
                    if isinstance(seq[0], torch.Tensor):
                        pad_val = torch.zeros_like(seq[0])
                    else:
                        pad_val = 0
                    seq = seq + [pad_val] * pad_len
                d[k] = seq
            elif isinstance(seq, torch.Tensor):
                if len(seq.shape) > 0:
                    n_pad = self.n - seq.shape[0]
                    seq_padded = torch.cat(
                        [seq, torch.zeros(n_pad, *seq.shape[1:], device=seq.device)],
                        dim=0,
                    )
                    d[k] = seq_padded
                else:
                    d[k] = seq
            else:
                raise TypeError(f"type is wrong'{k}': {type(seq)}")

        return d
