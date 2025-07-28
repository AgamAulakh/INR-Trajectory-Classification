#!/bin/bash

# setup
BASE_DIR=/home/user/path-to-save-all-checkpoints
IRR_DIR=${BASE_DIR}/irregular_sampling
REG_DIR=${BASE_DIR}/regular_sampling

# (run preprocess.py to generate the following data splits)
# these should point to datasplit with unpaired training set
IRR_FILE=/home/user/path-to-preprocessed-irregular-sampling-datasplit.pt
REG_FILE=/home/user/path-to-preprocessed-regular-sampling-datasplit.pt

# training initialization with only the unpaired training dataset (to conserve cache memory bc all subjs is unreasonable :) and unnecessary)
# simulating larger batch size for irregular sampling experiment
# to avoid needing to pad subject data for batch collation (different # of timepoints per subject)
python train_inr.py --n-input 4 --n-epoch 250 --batch-size 1 --simulate-batch-size 3 --lr 0.0001 --pos-encoding none --n-layers 8 --hidden-size 512 --fratio 0.9 --use-sep-spacetime --n-time-layers 5 --activation wire --wire-omega-0 10 --wire-sigma-0 30 --time-activation relu --report --use-interpol-extrapol --preprocessed-data-file ${IRR_FILE} --save-checkpoint-to-dir ${IRR_DIR} --save-checkpoints-at-epoch 10 50 100 250
python train_inr.py --n-input 4 --n-epoch 250 --batch-size 3 --lr 0.0001 --pos-encoding none --n-layers 8 --hidden-size 512 --fratio 0.9 --use-sep-spacetime --n-time-layers 5 --activation wire --wire-omega-0 10 --wire-sigma-0 30 --time-activation relu --report --use-interpol-extrapol --preprocessed-data-file ${REG_FILE} --save-checkpoint-to-dir ${REG_DIR} --save-checkpoints-at-epoch 10 50 100 250

