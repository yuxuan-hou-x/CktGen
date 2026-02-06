#!/bin/bash
# Source shared environment setup
. "$(dirname "$0")/../../setup_env.sh"


python -m train.train_evaluator \
--data_name ckt_bench_301 \
--data_fold_name CktBench301 \
--modelname evaluator_digin_nce_gde_pred \
--hidden_dim 512 \
--emb_dim 256 \
--latent_dim 256 \
--save_interval 100 \
--eval_interval 100 \
--batch_size 32 \
--epochs 100 \
--conditioned \
--contrastive \
--out_dir ./output/evaluator \
--exp_name evaluator_301 \