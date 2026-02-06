#!/bin/bash
# Source shared environment setup
. "$(dirname "$0")/../../setup_env.sh"


python -m test.test_auto_design \
--data_name ckt_bench_301 \
--data_fold_name CktBench301 \
--out_dir ./output/cktgen \
--exp_name cktgen_auto_design_301 \
--pretrained_eval_resume_pth ./checkpoints/evaluator/evaluator_301.pth \
--resume_pth ./checkpoints/cktgen/cktgen_cond_gen_301.pth \