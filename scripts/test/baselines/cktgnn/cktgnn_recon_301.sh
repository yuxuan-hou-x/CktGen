#!/bin/bash
# Source shared environment setup
. "$(dirname "$0")/../../../setup_env.sh"

python -m test.test_rand_gen \
--data_fold_name CktBench301 \
--data_name ckt_bench_301 \
--infer_batch_size 128 \
--out_dir ./output/baselines/cktgnn \
--exp_name cktgnn_recon_301 \
--resume_pth ./checkpoints/baselines/cktgnn/cktgnn_recon_rand_gen_301.pth \