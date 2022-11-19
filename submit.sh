#!/bin/bash

# Data locations
rna="/tuba/datasets/enformer/tf_datasets/AIBS_MOp_SMARTseq_BasenjiBins/"
atac="/home/emukamel/enformer/prepare_training_data/Li2021_mouse_snATAC_binsize128_BasenjiBins"
mc="/home/emukamel/enformer/prepare_training_data/CEMBA_mc_mincov5_smooth8_cg_ch"

sbatch_parms="--mem 32G -J enformer -p general_gpu_a5000,general_gpu_p6000"
trainnet_parms="--epochs 5000 \
                --targets_dir $rna \
                --out_activation linear \
                --model_architecture EpiEnformer_TwoStems"

# bash -c "sbatch $sbatch_parms ./trainnet_enformer.sh $trainnet_parms --run_id mcSeq2RNA_mse_2stems --use_sequence True --predictors_dir $atac"
#bash -c "sbatch $sbatch_parms ./trainnet_enformer.sh $trainnet_parms --run_id Atac2RNA_NoSeq_poisson_2stems --use_sequence True --loss poisson --log_target False --predictors_dir $atac"
#bash -c "sbatch $sbatch_parms ./trainnet_enformer.sh $trainnet_parms --run_id Atac2RNA_NoSeq_mseNoLog_2stems --use_sequence True --loss mse --log_target False --predictors_dir $atac"
#bash -c "sbatch $sbatch_parms ./trainnet_enformer.sh $trainnet_parms --run_id Atac2RNA_NoSeq_mseNoLog_2stems --nchannels 768 --use_sequence True --loss mse --log_target False --predictors_dir $atac --targets_dir $rna"

bash -c "sbatch $sbatch_parms ./trainnet_enformer.sh $trainnet_parms --run_id Atac2mc_NoSeq_mse_2stems --use_sequence False --loss mse --log_target False --predictors_dir $atac --targets_dir $mc"
