#!/bin/bash
# Change to project root (cktgen/)
cd ../../..

python -m test.test_rand_gen \
--data_fold_name CktBench301 \
--data_name ckt_bench_301 \
--modelname vae_cktarchi_recon_kl \
--infer_batch_size 128 \
--out_dir ./output/cktgen \
--exp_name cktgen_recon_301 \
--resume_pth ./checkpoints/cktgen/cktgen_recon_301.pth \