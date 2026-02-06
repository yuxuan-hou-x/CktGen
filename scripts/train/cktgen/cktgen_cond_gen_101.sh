#!/bin/bash
# Source shared environment setup
. "$(dirname "$0")/../../setup_env.sh"


python -m train.train_cktgen \
--modelname cktgen_cktarchi_kl_recon_align_nce_gde \
--data_name ckt_bench_101 \
--data_fold_name CktBench101 \
--save_interval 100 \
--eval_interval 600 \
--batch_size 32 \
--epochs 600 \
--lr 1e-4 \
--type_rate 0.5 \
--path_rate 0.05 \
--lambda_kl 1e-5 \
--temperature 1e-1 \
--out_dir ./output/cktgen \
--exp_name cktgen_cond_gen_101 \
--filter \
--conditioned \
--contrastive \
--vae \
--guide \
--pretrained_eval_resume_pth ./checkpoints/evaluator/evaluator_101.pth \