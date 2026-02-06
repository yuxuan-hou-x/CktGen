#!/bin/bash
# Source shared environment setup
. "$(dirname "$0")/../../../setup_env.sh"


python -m train.train_cvaegan \
--modelname cvaegan_cktarchi_kl_recon \
--data_name ckt_bench_301 \
--data_fold_name CktBench301 \
--save_interval 100 \
--eval_interval 100 \
--batch_size 32 \
--epochs 100 \
--lr 1e-4 \
--type_rate 0.7 \
--path_rate 0.07 \
--out_dir ./output/baselines/cvaegan \
--exp_name cvaegan_cond_gen_301 \
--conditioned \
--vae \
--pretrained_eval_resume_pth ./checkpoints/evaluator/evaluator_301.pth \