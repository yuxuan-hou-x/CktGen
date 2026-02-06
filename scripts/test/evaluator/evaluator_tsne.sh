#!/bin/bash
# Source shared environment setup
. "$(dirname "$0")/../../setup_env.sh"


python -m test.test_tsne \
--data_name ckt_bench_101 \
--data_fold_name CktBench101 \
--out_dir ./output/evaluator \
--exp_name evaluator_tsne \
--conditioned \
--vae \
--modelname evaluator_digin_nce_gde_pred \
--resume_pth ./checkpoints/evaluator/evaluator_101.pth \