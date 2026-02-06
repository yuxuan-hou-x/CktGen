#!/bin/bash
# Source shared environment setup
. "$(dirname "$0")/../../../setup_env.sh"


# CUDA_VISIBLE_DEVICES=1 
python -m train.train_cktgen \
--data_fold_name CktBench301 \
--data_name ckt_bench_301 \
--modelname cktgen_cktgnn_kl_recon_align_nce_gde \
--save_interval 100 \
--eval_interval 600 \
--epochs 600 \
--batch_size 64 \
--hid_dim 301 \
--out_dir ./output/baselines/cktgnn \
--exp_name cktgnn_cond_gen_301 \
--lambda_kl 1e-5 \
--eps_factor 1 \
--sized \
--filter \
--conditioned \
--contrastive \
--vae \
--guide \
--pretrained_eval_resume_pth ./checkpoints/evaluator/evaluator_301.pth \