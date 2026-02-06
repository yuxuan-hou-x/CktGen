#!/bin/bash
# Source shared environment setup
. "$(dirname "$0")/../../../setup_env.sh"


python -m train.train_ldt \
--modelname ldt_cktarchi_mse \
--data_name ckt_bench_301 \
--data_fold_name CktBench301 \
--save_interval 100 \
--eval_interval 50 \
--batch_size 32 \
--epochs 100 \
--lr 1e-4 \
--conditioned \
--vae \
--out_dir ./output/baselines/ldt \
--exp_name ldt_cond_gen_301 \
--cuda true \
--pretrained_eval_resume_pth ./checkpoints/evaluator/evaluator_301.pth \
--vae_pth ./checkpoints/cktgen/cktgen_recon_301.pth \