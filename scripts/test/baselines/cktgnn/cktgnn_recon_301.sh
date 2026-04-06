#!/bin/bash
# Change to project root (cktgen/)
cd ../../../..

python -m test.test_rand_gen \
--modelname vae_cktgnn_recon_kl \
--data_fold_name CktBench301 \
--data_name ckt_bench_301 \
--infer_batch_size 128 \
--out_dir ./output/baselines/cktgnn \
--exp_name cktgnn_recon_301 \
--resume_pth ./checkpoints/baselines/cktgnn/cktgnn_recon_301.pth \