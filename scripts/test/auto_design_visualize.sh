#!/bin/bash
# Change to project root (cktgen/)
cd ../..

python -m test.auto_design_visualize \
--modelname cktgen_cktarchi_kl_recon \
--data_name ckt_bench_101 \
--data_fold_name CktBench101 \
--out_dir ./output \
--exp_name auto_design_visualize \
--conditioned \