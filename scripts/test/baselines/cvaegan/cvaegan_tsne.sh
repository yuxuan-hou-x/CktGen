#!/bin/bash
# Change to project root (cktgen/)
cd ../../../..

python -m test.test_tsne \
--modelname cvaegan_cktarchi_kl_recon \
--data_name ckt_bench_101 \
--data_fold_name CktBench101 \
--out_dir ./output/baselines/cvaegan \
--exp_name cvaegan_tsne \
--conditioned \
--vae \
--resume_pth ./checkpoints/baselines/cvaegan/cvaegan_cond_gen_101.pth \