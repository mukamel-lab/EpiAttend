sbatch_parms="--mem 32G -J enformer -p general_gpu_a5000,general_gpu_p6000"
trainnet_parms="--epochs 5000 --model_architecture EpiEnformer_MLP"
mc="/tuba/datasets/enformer/tf_datasets/CEMBA_MOp_CGN_sm8"
atac="/tuba/datasets/enformer/tf_datasets/CEMBA_MOp_ATAC"
sbatch $sbatch_parms ./submit_slurm.sh trainnet_cellgeneralize.py $trainnet_parms --run_id test_MLP --use_sequence False --loss mse --log_target False --predictors_dir $atac --targets_dir $mc --out_activation sigmoid

tail slurm-*.out -f