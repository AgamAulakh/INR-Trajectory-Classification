#!/bin/bash

# setup
BASE_DIR=/home/user/path-to-save-all-checkpoints
IRR_DIR=${BASE_DIR}/irregular_sampling
REG_DIR=${BASE_DIR}/regular_sampling

# (run preprocess.py to generate the following data splits)
# these should point to datasplit with unpaired training set
IRR_FILE=/home/user/path-to-preprocessed-irregular-sampling-datasplit.pt
REG_FILE=/home/user/path-to-preprocessed-regular-sampling-datasplit.pt

# classification
# NOTE: may have to evaluate on testing data separately
#       ie. first train without loading test data,
#           then load a trained model from checkpoint, disable training/val dataloaders, and skip to testing in python script
# NOTE: batch size is 2 here, but we perform gradient accumulation for n_grad_step=10 (enabled by default) to simulate batch size of 20
python classify_images_sfcn.py --preprocessed-data-file ${IRR_FILE} --batch-size 2 --n-timepoints 5 --save-checkpoint-to-dir ${IRR_DIR}/sfcn/
python classify_images_sfcn.py --preprocessed-data-file ${REG_FILE} --batch-size 2 --n-timepoints 4 --save-checkpoint-to-dir ${REG_DIR}/sfcn/

# if testing separately, run the following after setting train/val datasets to none
python classify_images_sfcn.py --preprocessed-data-file ${IRR_FILE} --n-epoch 0 --batch-size 2 --n-timepoints 5 --save-checkpoint-to-dir ${IRR_DIR}/sfcn/ --load-checkpoint-from-file ${IRR_DIR}/sfcn/classify_images_sfcn_b2_e100.pt
python classify_images_sfcn.py --preprocessed-data-file ${REG_FILE} --n-epoch 0 --batch-size 2 --n-timepoints 4 --save-checkpoint-to-dir ${REG_DIR}/sfcn/ --load-checkpoint-from-file ${REG_DIR}/sfcn/classify_images_sfcn_b2_e100.pt
