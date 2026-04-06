#!/bin/bash
# Change to project root (cktgen/)
cd ../../..

python -m test.test_rand_gen \
--data_name ckt_bench_101 \
--data_fold_name CktBench101 \
--modelname evaluator_digin_nce_gde_pred \
--out_dir ./output/evaluator \
--exp_name evaluator_101 \
--resume_pth ./checkpoints/evaluator/evaluator_101.pth \