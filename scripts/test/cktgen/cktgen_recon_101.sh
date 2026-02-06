#!/bin/bash
# Source shared environment setup
. "$(dirname "$0")/../../setup_env.sh"


python -m test.test_rand_gen \
--modelname vae_cktarchi_recon_kl \
--data_fold_name CktBench101 \
--data_name ckt_bench_101 \
--infer_batch_size 128 \
--out_dir ./output/cktgen \
--exp_name cktgen_recon_101 \
--resume_pth ./checkpoints/cktgen/cktgen_recon_101.pth \