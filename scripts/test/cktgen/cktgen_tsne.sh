#!/bin/bash
# Change to project root (cktgen/)
cd ../../..

python -m test.test_tsne \
--data_name ckt_bench_101 \
--data_fold_name CktBench101 \
--out_dir ./output/cktgen \
--exp_name cktgen_tsne \
--conditioned \
--vae \
--modelname cktgen_cktarchi_kl_recon \
--resume_pth ./checkpoints/cktgen/cktgen_cond_gen_101.pth \