#!/bin/bash

# Data locations
rna="/tuba/datasets/enformer/tf_datasets/CEMBA_MOp_scRNA"
atac="/tuba/datasets/enformer/tf_datasets/CEMBA_MOp_ATAC"
mc="/tuba/datasets/enformer/tf_datasets/CEMBA_MOp_CGN_sm8"

sbatch_parms="--mem 32G -J enformer -p general_gpu_a5000,general_gpu_p6000"
trainnet_parms="--epochs 5000 \
                --model_architecture EpiEnformer_TwoStems"

# bash -c "sbatch $sbatch_parms ./trainnet_enformer.sh $trainnet_parms --run_id CEMBA_MOp_atac2mc --use_sequence False --loss mse --log_target False --predictors_dir $atac --targets_dir $mc"
# bash -c "sbatch $sbatch_parms ./trainnet_enformer.sh $trainnet_parms --run_id CEMBA_MOp_mc2rna --use_sequence False --loss mse --log_target True --predictors_dir $mc --targets_dir $rna --out_activation linear "

bash -c "sbatch $sbatch_parms ./trainnet_enformer.sh $trainnet_parms --run_id CEMBA_MOp_mcatac2rna --use_sequence False --loss mse --log_target False --predictors_dir $mc,$atac --targets_dir $rna --out_activation softplus "
