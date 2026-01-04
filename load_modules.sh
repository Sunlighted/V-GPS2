#!/bin/bash
# load_modules.sh

deactivate
module load StdEnv/2023 gcc/12.3
module load opencv/4.8.1
module load cuda/12.2
module load cudnn/8.9.5.29
module load python/3.10
module load scipy-stack

source VGPS/bin/activate