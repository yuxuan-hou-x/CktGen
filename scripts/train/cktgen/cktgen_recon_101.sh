#!/bin/bash
# Source shared environment setup
. "$(dirname "$0")/../../setup_env.sh"


python -m train.train_vae \
--data_fold_name CktBench101 \
--data_name ckt_bench_101 \
--modelname vae_cktarchi_recon_kl \
--save_interval 100 \
--eval_interval 100 \
--epochs 400 \
--lr 1e-4 \
--batch_size 32 \
--infer_batch_size 128 \
--lambda_kl 5e-3 \
--block_size 9 \
--dropout_rate 0.3 \
--type_rate 0.5 \
--path_rate 0.05 \
--out_dir ./output/cktgen \
--exp_name cktgen_recon_101 \
--vae \