#!/bin/bash
# Change to project root (cktgen/)
cd ../../../..

python -m test.test_auto_design \
--data_name ckt_bench_301 \
--data_fold_name CktBench301 \
--out_dir ./output/baselines/cktgnn \
--exp_name cktgnn_auto_design_301 \
--pretrained_eval_resume_pth ./checkpoints/evaluator/evaluator_301.pth \
--resume_pth ./checkpoints/baselines/cktgnn/cktgnn_cond_gen_301.pth \