#!/bin/bash
# Source shared environment setup
. "$(dirname "$0")/../../../setup_env.sh"


python -m train.train_cktgen \
--data_fold_name CktBench101 \
--data_name ckt_bench_101 \
--modelname cktgen_pace_kl_recon_align_nce_gde \
--batch_size 32 \
--save_interval 100 \
--eval_interval 600 \
--epochs 600 \
--lr 1e-4 \
--lambda_kl 1e-5 \
--emb_dim 32 \
--device_size_embed_dim 8 \
--hidden_dim 96 \
--fc_hidden 32 \
--latent_dim 96 \
--dropout 0.15 \
--num_layers 3 \
--num_heads 8 \
--eps_factor 1 \
--out_dir ./output/baselines/pace \
--exp_name pace_cond_gen_101 \
--filter \
--conditioned \
--contrastive \
--vae \
--guide \
--pretrained_eval_resume_pth ./checkpoints/evaluator/evaluator_101.pth \