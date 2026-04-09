#!/bin/bash
# Change to project root (cktgen/)
cd ../../..

python -m test.test_evaluator \
--data_name ckt_bench_301 \
--data_fold_name CktBench301 \
--modelname evaluator_digin_nce_gde_pred \
--out_dir ./output/evaluator \
--exp_name evaluator_301 \
--resume_pth ./checkpoints/evaluator/evaluator_301.pth \
