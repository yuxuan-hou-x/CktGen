#!/bin/bash
# Source shared environment setup
. "$(dirname "$0")/../setup_env.sh"

python -m test.auto_design_visualize \
--modelname cktgen_cktarchi_kl_recon \
--data_name ckt_bench_101 \
--data_fold_name CktBench101 \
--out_dir ./output \
--exp_name auto_design_visualize \
--conditioned \