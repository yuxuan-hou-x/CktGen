#!/bin/bash
# Source shared environment setup
. "$(dirname "$0")/../../../setup_env.sh"


python -m train.train_vae \
--data_fold_name CktBench101 \
--data_name ckt_bench_101 \
--modelname vae_pace_recon_kl \
--batch_size 32 \
--save_interval 100 \
--eval_interval 100 \
--epochs 100 \
--lr 1e-4 \
--lambda_kl 5e-3 \
--emb_dim 32 \
--device_size_embed_dim 8 \
--hidden_dim 96 \
--fc_hidden 32 \
--latent_dim 96 \
--dropout 0.15 \
--num_layers 3 \
--num_heads 8 \
--eps_factor 0.01 \
--out_dir ./output/baselines/pace \
--exp_name pace_recon_101 \
--vae \