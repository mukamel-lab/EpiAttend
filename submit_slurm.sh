#!/bin/bash

source $HOME/.bashrc
module load cuda/11.2 
export TF_XLA_FLAGS=--tf_xla_enable_xla_devices

./mypython "$@"

exit
