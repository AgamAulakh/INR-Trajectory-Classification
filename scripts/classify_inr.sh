#!/bin/bash

# setup
BASE_DIR=/home/user/path-to-save-all-checkpoints
IRR_DIR=${BASE_DIR}/irregular_sampling
REG_DIR=${BASE_DIR}/regular_sampling

# (run preprocess.py to generate the following data splits)
# these should point to datasplit with unpaired training set (not alltrain)
IRR_FILE=/home/user/path-to-preprocessed-irregular-sampling-datasplit.pt
REG_FILE=/home/user/path-to-preprocessed-regular-sampling-datasplit.pt

# If trained on different # of adaptation "epochs," change ADAPT_EPOCHS
# For now, we only classify after the 100th update
ADAPT_EPOCHS=(100)
for epoch in "${ADAPT_EPOCHS[@]}"; do
    echo REGULAR SAMPLING
    python classify_inr.py --load-inrs-from-dir ${REG_DIR}/${epoch}epochs --preprocessed-data-file ${REG_FILE} --save-checkpoint-to-dir ${REG_DIR}/${epoch}epochs/ --batch-size 20 --hidden-dims 512 1024 2048 --embed-dim 2048 --use-space-layers
    python classify_inr.py --load-inrs-from-dir ${REG_DIR}/${epoch}epochs --preprocessed-data-file ${REG_FILE} --save-checkpoint-to-dir ${REG_DIR}/${epoch}epochs/ --batch-size 20 --hidden-dims 512 1024 2048 --embed-dim 2048 --use-time-layers
    python classify_inr.py --load-inrs-from-dir ${REG_DIR}/${epoch}epochs --preprocessed-data-file ${REG_FILE} --save-checkpoint-to-dir ${REG_DIR}/${epoch}epochs/ --batch-size 20 --hidden-dims 512 1024 2048 --embed-dim 2048 --use-time-layers --use-combined-layers
    python classify_inr.py --load-inrs-from-dir ${REG_DIR}/${epoch}epochs --preprocessed-data-file ${REG_FILE} --save-checkpoint-to-dir ${REG_DIR}/${epoch}epochs/ --batch-size 20 --hidden-dims 512 1024 2048 --embed-dim 2048 --use-time-layers --use-space-layers --use-combined-layers
    python classify_inr.py --load-inrs-from-dir ${REG_DIR}/${epoch}epochs --preprocessed-data-file ${REG_FILE} --save-checkpoint-to-dir ${REG_DIR}/${epoch}epochs/ --batch-size 20 --hidden-dims 512 1024 2048 --embed-dim 2048 --use-space-layers --use-combined-layers
    python classify_inr.py --load-inrs-from-dir ${REG_DIR}/${epoch}epochs --preprocessed-data-file ${REG_FILE} --save-checkpoint-to-dir ${REG_DIR}/${epoch}epochs/ --batch-size 20 --hidden-dims 512 1024 2048 --embed-dim 2048 --use-space-layers --use-time-layers
    python classify_inr.py --load-inrs-from-dir ${REG_DIR}/${epoch}epochs --preprocessed-data-file ${REG_FILE} --save-checkpoint-to-dir ${REG_DIR}/${epoch}epochs/ --batch-size 20 --hidden-dims 512 1024 2048 --embed-dim 2048 --use-combined-layers

    echo IRREGULAR
    python classify_inr.py --load-inrs-from-dir ${IRR_DIR}/${epoch}epochs --preprocessed-data-file ${IRR_FILE} --save-checkpoint-to-dir ${IRR_DIR}/${epoch}epochs/ --batch-size 20 --hidden-dims 512 1024 2048 --embed-dim 2048 --use-space-layers
    python classify_inr.py --load-inrs-from-dir ${IRR_DIR}/${epoch}epochs --preprocessed-data-file ${IRR_FILE} --save-checkpoint-to-dir ${IRR_DIR}/${epoch}epochs/ --batch-size 20 --hidden-dims 512 1024 2048 --embed-dim 2048 --use-time-layers
    python classify_inr.py --load-inrs-from-dir ${IRR_DIR}/${epoch}epochs --preprocessed-data-file ${IRR_FILE} --save-checkpoint-to-dir ${IRR_DIR}/${epoch}epochs/ --batch-size 20 --hidden-dims 512 1024 2048 --embed-dim 2048 --use-time-layers --use-combined-layers
    python classify_inr.py --load-inrs-from-dir ${IRR_DIR}/${epoch}epochs --preprocessed-data-file ${IRR_FILE} --save-checkpoint-to-dir ${IRR_DIR}/${epoch}epochs/ --batch-size 20 --hidden-dims 512 1024 2048 --embed-dim 2048 --use-time-layers --use-space-layers --use-combined-layers
    python classify_inr.py --load-inrs-from-dir ${IRR_DIR}/${epoch}epochs --preprocessed-data-file ${IRR_FILE} --save-checkpoint-to-dir ${IRR_DIR}/${epoch}epochs/ --batch-size 20 --hidden-dims 512 1024 2048 --embed-dim 2048 --use-space-layers --use-combined-layers
    python classify_inr.py --load-inrs-from-dir ${IRR_DIR}/${epoch}epochs --preprocessed-data-file ${IRR_FILE} --save-checkpoint-to-dir ${IRR_DIR}/${epoch}epochs/ --batch-size 20 --hidden-dims 512 1024 2048 --embed-dim 2048 --use-space-layers --use-time-layers
    python classify_inr.py --load-inrs-from-dir ${IRR_DIR}/${epoch}epochs --preprocessed-data-file ${IRR_FILE} --save-checkpoint-to-dir ${IRR_DIR}/${epoch}epochs/ --batch-size 20 --hidden-dims 512 1024 2048 --embed-dim 2048 --use-combined-layers

done