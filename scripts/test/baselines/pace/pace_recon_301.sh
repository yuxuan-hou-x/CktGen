#!/bin/bash
# Source shared environment setup
. "$(dirname "$0")/../../../setup_env.sh"

python -m test.test_rand_gen \
--modelname vae_pace_recon_kl \
--data_name ckt_bench_301 \
--data_fold_name CktBench301 \
--infer_batch_size 128 \
--out_dir ./output/baselines/pace \
--exp_name pace_recon_301 \
--resume_pth ./checkpoints/baselines/pace/pace_recon_rand_gen_301.pth \