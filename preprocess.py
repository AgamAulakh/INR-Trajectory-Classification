import os
import glob
import random
import argparse

import numpy as np
from sklearn.model_selection import train_test_split

from construct_dataset import DataSplit

def expand_adhealthy_to_data_dicts(adhealthy_dicts, is_paired=True):
    if adhealthy_dicts is None:
        return None

    expanded_dicts = []
    for idx, adhealthy_subj in enumerate(adhealthy_dicts):
        subj = adhealthy_subj["subject"]
        subj_idx = adhealthy_subj["subject_idx"]
        age = adhealthy_subj["age"]

        if is_paired:
            expanded_dicts.append({
                "image": adhealthy_subj["healthy_images"],
                "subject": f"{subj}_healthy",
                "subject_idx": subj_idx*2,
                "age": age,
                "fake_traj": False,
            })
            expanded_dicts.append({
                "image": adhealthy_subj["adlike_images"],
                "subject": f"{subj}_adlike",
                "subject_idx": subj_idx*2+1,
                "age": age,
                "fake_traj": True,
            })
        else:
            expanded_dicts.append({
                "image": adhealthy_subj["adlike_images"] if idx%2 else adhealthy_subj["healthy_images"],
                "subject": f"{subj}_adlike" if idx%2 else f"{subj}_healthy" ,
                "subject_idx": subj_idx*2+1,
                "age": age,
                "fake_traj": True if idx%2 else False,
            })


    return expanded_dicts


def create_train_val_test_split_traj_sim(
    data_dir,
    seed,
    split=(1.0, 0.0, 0.0),
    max_subjs=500,
    is_sampling_const=False,
):
    assert sum(split) == 1.0

    nifti_files = glob.glob(os.path.join(data_dir, "sub*", "[ADlikeHealthy]*", "*.nii.gz"))
    nifti_files.sort()

    subj_to_idx = {}
    subj_ages_to_sample = {}
    data_dicts = []
    age_list = []
    split_to_save = DataSplit(age_1p=0, age_99p=0, is_age_clipped=False)

    for file in nifti_files:
        basename = os.path.basename(file)
        subj = [s for s in basename.split("_") if s.startswith("subj")][0][4:]
        age = int([s for s in basename.split("_") if s.endswith(".gz")][0][:2])
        age_list.append(age)

        if len(data_dicts) < max_subjs:
            if subj not in subj_to_idx:
                subj_to_idx[subj] = len(data_dicts)
                # Note: this dict is not compatible with DataSplit
                #       need to break apart healthy and adlike after splits are made
                data_dicts.append(
                    {
                        "healthy_images": [],
                        "adlike_images": [],
                        "age": [],
                        "subject": subj,
                        "subject_idx": subj_to_idx[subj],
                        "fake_traj": False,
                    }
                )
                subj_ages_to_sample[subj] = sorted(random.sample(range(50, 76), k=random.randint(3, 5)))

        if is_sampling_const:
            # constant sampling:
            if subj in subj_to_idx and age in (50,58,67,75):
                # add data in ascending order
                i = len([x for x in data_dicts[subj_to_idx[subj]]["age"] if x < age])
                if age not in data_dicts[subj_to_idx[subj]]["age"]:
                    data_dicts[subj_to_idx[subj]]["age"].insert(i, age)
                if "Healthy" in basename:
                    data_dicts[subj_to_idx[subj]]["healthy_images"].insert(i, file)
                else:
                    data_dicts[subj_to_idx[subj]]["adlike_images"].insert(i, file)
        else:
            # pick 3-5 random ages form 50-75 
            if subj in subj_to_idx and age in subj_ages_to_sample[subj]:
                i = len([x for x in data_dicts[subj_to_idx[subj]]["age"] if x < age])
                if age not in data_dicts[subj_to_idx[subj]]["age"]:
                    data_dicts[subj_to_idx[subj]]["age"].insert(i, age)
                if "Healthy" in basename:
                    data_dicts[subj_to_idx[subj]]["healthy_images"].insert(i, file)
                else:
                    data_dicts[subj_to_idx[subj]]["adlike_images"].insert(i, file)


    split_to_save.age_1p = np.percentile(age_list, 1)
    split_to_save.age_99p = np.percentile(age_list, 99)

    # stratify all subjects
    if split[2] > 0:
        train_val_split, test_split = train_test_split(
            data_dicts,
            test_size=split[2],
            random_state=seed,
        )
    else:
        train_val_split = data_dicts
        test_split = None

    val_size = split[1] / (split[0] + split[1])
    if val_size > 0.0:
        train_split, val_split = train_test_split(
            train_val_split,
            test_size=split[1] / (split[0] + split[1]),
            random_state=seed,
        )
    else:
        # assume we dont use validation data rn
        train_split = train_val_split
        val_split = None

    # break down "healthy_images" and "adlike_images" into two fake subjects with "image" list
    split_to_save.train_data_dicts = expand_adhealthy_to_data_dicts(train_split, is_paired=False)
    split_to_save.val_data_dicts = expand_adhealthy_to_data_dicts(val_split)
    split_to_save.test_data_dicts = expand_adhealthy_to_data_dicts(test_split)
    split_to_save.normalize_ages_over_all_data()

    split_to_save.save(
        f"data_splits/ode_2x500subjs/traj_sim_{max_subjs*2}subjs_trainUNPAIRED_t60v20t60_constSample{is_sampling_const}.pt"
    )

    # also save all patients in train data
    all_data_split = train_split + val_split + test_split
    split_to_save.train_data_dicts = expand_adhealthy_to_data_dicts(all_data_split)
    split_to_save.val_data_dicts = None
    split_to_save.test_data_dicts = None
    split_to_save.save(
        f"data_splits/ode_2x500subjs/traj_sim_{max_subjs*2}subjs_alltrain_constSample{is_sampling_const}.pt"
    )

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../Data/gen_images/",
        help="Directory where data will be loaded from",
    )
    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = get_args()
    print(f"accepted arguments: {args}")
    if unknown:
        print(f"ignored arguments: {unknown}")

    create_train_val_test_split_traj_sim(args.data_dir, args.seed)
