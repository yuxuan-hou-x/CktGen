#!/bin/bash
# Source shared environment setup
. "$(dirname "$0")/../../../setup_env.sh"

python -m test.test_tsne \
--modelname cktgen_pace_kl_recon \
--data_name ckt_bench_101 \
--data_fold_name CktBench101 \
--out_dir ./output/baselines/pace \
--exp_name pace_tsne \
--conditioned \
--vae \
--resume_pth ./checkpoints/baselines/pace/pace_cond_gen_101.pth \