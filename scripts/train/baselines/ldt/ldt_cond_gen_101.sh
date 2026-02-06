#!/bin/bash
# Source shared environment setup
. "$(dirname "$0")/../../../setup_env.sh"


python -m train.train_ldt \
--modelname ldt_cktarchi_mse \
--data_name ckt_bench_101 \
--data_fold_name CktBench101 \
--save_interval 100 \
--eval_interval 100 \
--batch_size 32 \
--epochs 600 \
--lr 1e-4 \
--conditioned \
--vae \
--out_dir ./output/baselines/ldt \
--exp_name ldt_cond_gen_101 \
--cuda true \
--vae_pth ./checkpoints/cktgen/cktgen_recon_101.pth \
--pretrained_eval_resume_pth ./checkpoints/evaluator/evaluator_101.pth \