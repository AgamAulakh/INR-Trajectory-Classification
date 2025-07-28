#!/bin/bash

# setup
BASE_DIR=/home/user/path-to-save-all-checkpoints
IRR_DIR=${BASE_DIR}/irregular_sampling
REG_DIR=${BASE_DIR}/regular_sampling

# (run preprocess.py to generate the following data splits)
# these should point to datasplit with unpaired training set
IRR_FILE=/home/user/path-to-preprocessed-irregular-sampling-datasplit.pt
REG_FILE=/home/user/path-to-preprocessed-regular-sampling-datasplit.pt

# these should point to datasplit with all data in training set (train = original unpaired training set + val set + test set)
IRR_ALL_TRAIN_FILE=/home/user/path-to-preprocessed-ALLTRAIN-irregular-sampling-datasplit.pt
REG_ALL_TRAIN_FILE=/home/user/path-to-preprocessed-ALLTRAIN-regular-sampling-datasplit.pt

# path to the initializations produced by inr_init.sh
IRR_INIT_FILE=${IRR_DIR}/name-of-irregular-sampling-initialization.pt
REG_INIT_FILE=${REG_DIR}/name-of-regular-sampling-initialization.pt

# adapt to ALL subjects with irregular sampling one-by-one (each should now be in the "training" set which we will use for adaptation)
# can save at different # of iterations, but in this paper, we only adapt for 100 gradient steps ("epochs")
EPOCHS=(100)
for subj in $(seq 0 499); do
    padded_subj=$(printf "%04d" $((subj + 1)))
    subj_idx_healthy=$(( 2 * subj ))
    subj_idx_adlike=$(( 2 * subj + 1 ))

    for epoch in "${EPOCHS[@]}"; do
        cur_epoch=100
        python train_inr.py --n-input 4 --n-epoch ${epoch} --batch-size 1 --lr 0.001 --pos-encoding none --n-layers 8 --hidden-size 512 --fratio 0.9 --use-sep-spacetime --n-time-layers 5 --activation wire --wire-omega-0 10 --wire-sigma-0 30 --time-activation relu --report --use-interpol-extrapol --subj $subj_idx_healthy --adapt-to-subj --preprocessed-data-file ${IRR_ALL_TRAIN_FILE} --load-checkpoint-from-file ${IRR_INIT_FILE} --save-checkpoint-to-dir ${IRR_DIR}/${cur_epoch}epochs/
        python train_inr.py --n-input 4 --n-epoch ${epoch} --batch-size 1 --lr 0.001 --pos-encoding none --n-layers 8 --hidden-size 512 --fratio 0.9 --use-sep-spacetime --n-time-layers 5 --activation wire --wire-omega-0 10 --wire-sigma-0 30 --time-activation relu --report --use-interpol-extrapol --subj $subj_idx_adlike --adapt-to-subj --preprocessed-data-file ${IRR_ALL_TRAIN_FILE} --load-checkpoint-from-file ${IRR_INIT_FILE} --save-checkpoint-to-dir ${IRR_DIR}/${cur_epoch}epochs/
    done
done

# adapt to ALL subjects with regular sampling one-by-one
for subj in $(seq 0 499); do
    prev_epoch_diff=100
    padded_subj=$(printf "%04d" $((subj + 1)))
    subj_idx_healthy=$(( 2 * subj ))
    subj_idx_adlike=$(( 2 * subj + 1 ))

    for epoch in "${EPOCHS[@]}"; do
        cur_epoch=100
        python train_inr.py --n-input 4 --n-epoch ${epoch} --batch-size 1 --lr 0.001 --pos-encoding none --n-layers 8 --hidden-size 512 --fratio 0.9 --use-sep-spacetime --n-time-layers 5 --activation wire --wire-omega-0 10 --wire-sigma-0 30 --time-activation relu --report --use-interpol-extrapol --subj $subj_idx_healthy --adapt-to-subj --preprocessed-data-file ${REG_ALL_TRAIN_FILE} --load-checkpoint-from-file ${REG_INIT_FILE} --save-checkpoint-to-dir ${REG_DIR}/${cur_epoch}epochs/
        python train_inr.py --n-input 4 --n-epoch ${epoch} --batch-size 1 --lr 0.001 --pos-encoding none --n-layers 8 --hidden-size 512 --fratio 0.9 --use-sep-spacetime --n-time-layers 5 --activation wire --wire-omega-0 10 --wire-sigma-0 30 --time-activation relu --report --use-interpol-extrapol --subj $subj_idx_adlike --adapt-to-subj --preprocessed-data-file ${REG_ALL_TRAIN_FILE} --load-checkpoint-from-file ${REG_INIT_FILE} --save-checkpoint-to-dir ${REG_DIR}/${cur_epoch}epochs/
    done
done

# calc image reconstruction stats
python visualize_inr.py --preprocessed-data-file $IRR_FILE --load-inrs-from-dir ${IRR_DIR}/100epochs
python visualize_inr.py --preprocessed-data-file $REG_FILE --load-inrs-from-dir ${REG_DIR}/100epochs
