#!/bin/bash
# Source shared environment setup
. "$(dirname "$0")/../../setup_env.sh"


python -m train.train_evaluator \
--data_name ckt_bench_301 \
--data_fold_name CktBench301 \
--modelname evaluator_digin_nce_gde_pred \
--out_dir ./output/evaluator \
--exp_name evaluator_301 \
--resume_pth ./checkpoints/evaluator/evaluator_301.pth \