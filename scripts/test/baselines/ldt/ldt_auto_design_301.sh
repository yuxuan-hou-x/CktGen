#!/bin/bash
# Source shared environment setup
. "$(dirname "$0")/../../../setup_env.sh"

python -m test.test_auto_design \
--modelname ldt_cktarchi_mse \
--data_name ckt_bench_301 \
--data_fold_name CktBench301 \
--out_dir ./output/baselines/ldt \
--exp_name ldt_auto_design_301 \
--pretrained_eval_resume_pth ./checkpoints/evaluator/evaluator_301.pth \
--resume_pth ./checkpoints/baselines/ldt/ldt_cond_gen_301.pth \