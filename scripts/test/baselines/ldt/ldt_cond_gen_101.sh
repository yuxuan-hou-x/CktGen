#!/bin/bash
# Change to project root (cktgen/)
cd ../../../..

python -m test.test_cond_gen \
--modelname ldt_cktarchi_mse \
--data_name ckt_bench_101 \
--data_fold_name CktBench101 \
--out_dir ./output/baselines/ldt \
--exp_name ldt_cond_gen_101 \
--pretrained_eval_resume_pth ./checkpoints/evaluator/evaluator_101.pth \
--resume_pth ./checkpoints/baselines/ldt/ldt_cond_gen_101.pth \