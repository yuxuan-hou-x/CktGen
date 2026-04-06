#!/bin/bash
# Change to project root (cktgen/)
cd ../../../..

python -m test.test_tsne \
--modelname ldt_cktarchi_mse \
--data_name ckt_bench_101 \
--data_fold_name CktBench101 \
--out_dir ./output/baselines/ldt \
--exp_name ldt_tsne \
--conditioned \
--resume_pth ./checkpoints/baselines/ldt/ldt_cond_gen_101.pth \