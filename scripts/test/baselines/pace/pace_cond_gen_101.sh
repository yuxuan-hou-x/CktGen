#!/bin/bash
# Source shared environment setup
. "$(dirname "$0")/../../../setup_env.sh"

python -m test.test_cond_gen \
--modelname cktgen_pace_kl_recon_align_nce_gde \
--data_name ckt_bench_101 \
--data_fold_name CktBench101 \
--out_dir ./output/baselines/pace \
--exp_name pace_cond_gen_101 \
--pretrained_eval_resume_pth ./checkpoints/evaluator/evaluator_101.pth \
--resume_pth ./checkpoints/baselines/pace/pace_cond_gen_101.pth \