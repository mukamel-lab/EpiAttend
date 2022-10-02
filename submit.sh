#!/bin/bash

sbatch_parms="--mem 32G -J enformer -p general_gpu_a5000,general_gpu_p6000,general_gpu_k80"
trainnet_parms="--epochs 5000 --targets_dir /tuba/datasets/enformer/tf_datasets/AIBS_MOp_SMARTseq_BasenjiBins/ --loss mse --out_activation linear --log_target True --side_trunk_depth 3"

bash -c "sbatch $sbatch_parms ./trainnet_enformer.sh $trainnet_parms \
     --run_id mc2RNA_NoSeq_logmse  \
     --use_sequence False \
     --predictors_dir /home/emukamel/enformer/prepare_training_data/CEMBA_mc_mincov5_smooth8_cg_ch"

bash -c "sbatch $sbatch_parms ./trainnet_enformer.sh $trainnet_parms \
     --run_id Atac2RNA_NoSeq_logmse  \
     --use_sequence False \
     --predictors_dir /home/emukamel/enformer/prepare_training_data/CEMBA_mc_mincov5_smooth8_cg_ch"

bash -c "sbatch $sbatch_parms ./trainnet_enformer.sh $trainnet_parms \
     --run_id mcSeq2RNA_logmse  \
     --use_sequence True \
     --predictors_dir /home/emukamel/enformer/prepare_training_data/CEMBA_mc_mincov5_smooth8_cg_ch"

bash -c "sbatch $sbatch_parms ./trainnet_enformer.sh $trainnet_parms \
     --run_id AtacSeq2RNA_logmse  \
     --use_sequence True \
     --predictors_dir /home/emukamel/enformer/prepare_training_data/CEMBA_mc_mincov5_smooth8_cg_ch"

# bash -c "sbatch $sbatch_parms ./trainnet_enformer.sh $trainnet_parms \
#     --run_id Atac2RNA_NoSeq_v2  \
# 	--use_sequence False \
#     --predictors_dir /home/emukamel/enformer/prepare_training_data/Li2021_mouse_snATAC_binsize128_BasenjiBins"

# bash -c "sbatch $sbatch_parms ./trainnet_enformer.sh $trainnet_parms \
#     --run_id AtacSeq2RNA_v2  \
# 	--use_sequence True \
#     --predictors_dir /home/emukamel/enformer/prepare_training_data/Li2021_mouse_snATAC_binsize128_BasenjiBins"

# bash -c "sbatch $sbatch_parms ./trainnet_enformer.sh $trainnet_parms \
#     --run_id mcSeq2RNA_v2  \
# 	--use_sequence True \
#     --predictors_dir /home/emukamel/enformer/prepare_training_data/CEMBA_mc_mincov5_smooth8_cg_ch"

