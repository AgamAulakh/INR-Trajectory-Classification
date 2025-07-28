import glob
import os
from typing import Dict, List, OrderedDict

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from inr.network import Network
from utils import get_mlp_params_as_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class INRDataset(Dataset):
    def __init__(
        self,
        data_dict: List[Dict],
        checkpoint_dir: str,
        net_config_file: str,
        use_space_layers: bool,
        use_time_layers: bool,
        use_combined_layers: bool,
        is_shuffle_cols_enable: bool,
    ):
        self.data = []
        self.subj_data = data_dict
        self.use_space_only_layers = use_space_layers
        self.use_time_only_layers = use_time_layers
        self.use_combined_layers = use_combined_layers
        self.is_shuffle_cols_enable = is_shuffle_cols_enable

        assert os.path.isdir(checkpoint_dir)
        inr_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
        inr_files.sort()

        for inr_file in inr_files:
            basename = os.path.basename(inr_file)
            for subj in self.subj_data:
                # silly - going through all data
                # if basename.startswith(f"subj{subj["subject_idx"]+1}_"):
                if basename.startswith(f"train_{subj['subject']}_"):
                    self.data.append(
                        {
                            "file": inr_file,
                            "subject": "subj" + subj["subject"],
                            "subject_idx": subj["subject_idx"],
                            "fake_traj": subj["fake_traj"],
                        }
                    )
                    break

        # NOTE: not using this anymore
        assert os.path.isfile(net_config_file)
        with open(net_config_file, "r") as f:
            self.net_config = yaml.safe_load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        subj = self.data[idx]

        inr_ckpt = torch.load(subj["file"], map_location="cpu", weights_only=True)
        inr_sd = inr_ckpt["network_state_dict"]
        feat_dim = 512
        layers_to_ignore = ["layers.0", "time_layers.0"]

        # remove unnecessary parameters
        if not self.use_space_only_layers:
            for i in range(5):
                layers_to_ignore.append(f"layers.{i}")
        if not self.use_time_only_layers:
            layers_to_ignore.append("time_layers")
        if not self.use_combined_layers:
            for i in range(5, 8):
                layers_to_ignore.append(f"layers.{i}")
        inr_sd_filtered = OrderedDict()
        for k, v in inr_sd.items():
            if v.shape in {(feat_dim,), (feat_dim, feat_dim)} and not any(
                k.startswith(x) for x in layers_to_ignore
            ):
                inr_sd_filtered[k] = v

        if self.is_shuffle_cols_enable:
            # shuffle v shuch that the inputs of ALL weights matrices are randomly permuted
            perm = torch.randperm(feat_dim)
            for k, v in inr_sd_filtered.items():
                if v.shape == (feat_dim, feat_dim):
                    inr_sd_filtered[k] = v[:, perm]

        # concatenate weights and biases
        inr_params = [
            v if v.ndim == 2 else v.unsqueeze(0)
            for k, v in inr_sd_filtered.items()
            if "weight" in k or "bias" in k
        ]
        inr_mat = torch.cat(inr_params, dim=0)
        return {
            "inr_mat": inr_mat,
            "subject": subj["subject"],
            "subject_idx": subj["subject_idx"],
            "fake_traj": subj["fake_traj"],
        }

    def collate(self, batch):
        inr_mats = [item["inr_mat"] for item in batch]
        subjects = [item["subject"] for item in batch]
        subject_idxs = [item["subject_idx"] for item in batch]
        fake_trajs = [item["fake_traj"] for item in batch]

        return (
            torch.stack(inr_mats).to(device),
            subjects,
            torch.tensor(subject_idxs, dtype=torch.int, device=device),
            torch.tensor(fake_trajs, dtype=torch.long, device=device),
        )


def construct_inr_dataloader(
    data_dict,
    checkpoint_dir,
    net_config_file,
    use_space_layers=True,
    use_time_layers=True,
    use_combined_layers=True,
    batch_size=5,
    shuffle_inrs=True,
    shuffle_cols=False,
):
    dataset = INRDataset(
        data_dict,
        checkpoint_dir,
        net_config_file,
        use_space_layers,
        use_time_layers,
        use_combined_layers,
        shuffle_cols,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_inrs,
        collate_fn=dataset.collate,
        num_workers=0,
        pin_memory=False,
    )
