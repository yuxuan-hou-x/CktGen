#!/bin/bash
# Source shared environment setup
. "$(dirname "$0")/../../../setup_env.sh"

python -m test.test_cond_gen \
--data_fold_name CktBench301 \
--data_name ckt_bench_301 \
--out_dir ./output/baselines/cktgnn \
--exp_name cktgnn_cond_gen_301 \
--pretrained_eval_resume_pth ./checkpoints/evaluator/evaluator_301.pth \
--resume_pth ./checkpoints/baselines/cktgnn/cktgnn_cond_gen_301.pth \