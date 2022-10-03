#!/bin/bash

#source /home/emukamel/emukamel/enformer/venv/bin/activate
source $HOME/.bashrc
# conda activate tf
module load cuda/11.2 # Note: 11.2 is not working
export TF_XLA_FLAGS=--tf_xla_enable_xla_devices

./mypython ./trainnet_enformer.py "$@" 

exit

