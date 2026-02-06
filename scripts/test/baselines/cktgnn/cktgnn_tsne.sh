#!/bin/bash
# Source shared environment setup
. "$(dirname "$0")/../../../setup_env.sh"

python -m test.test_tsne \
--data_name ckt_bench_101 \
--data_fold_name CktBench101 \
--out_dir ./output/baselines/cktgnn \
--exp_name cktgnn_tsne \
--conditioned \
--vae \
--resume_pth ./checkpoints/baselines/cktgnn/cktgnn_cond_gen_101.pth \