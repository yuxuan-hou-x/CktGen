#!/bin/bash
# Change to project root (cktgen/)
cd ../../../..

python -m test.test_cond_gen \
--modelname cvaegan_cktarchi_kl_recon \
--data_name ckt_bench_101 \
--data_fold_name CktBench101 \
--out_dir ./output/baselines/cvaegan \
--exp_name cvaegan_cond_gen_101 \
--pretrained_eval_resume_pth ./checkpoints/evaluator/evaluator_101.pth \
--resume_pth ./checkpoints/baselines/cvaegan/cvaegan_cond_gen_101.pth \