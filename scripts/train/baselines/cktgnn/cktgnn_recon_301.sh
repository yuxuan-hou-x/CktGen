#!/bin/bash
# Source shared environment setup
. "$(dirname "$0")/../../setup_env.sh"


python -m train.train_vae \
--data_fold_name CktBench301 \
--data_name ckt_bench_301 \
--modelname vae_cktgnn_recon_kl \
--save_interval 100 \
--eval_interval 300 \
--epochs 300 \
--batch_size 64 \
--hid_dim 301 \
--out_dir ./output/baselines/cktgnn \
--exp_name cktgnn_recon_301 \
--lambda_kl 5e-3 \
--eps_factor 0.01 \
--sized \
--vae \