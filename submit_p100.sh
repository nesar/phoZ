#!/bin/sh
#SBATCH -p cp100

echo [$SECONDS] setting up environment
#export KERAS_BACKEND=tensorflow

conda activate tf_gpu_14

#srun -p cp100 python mdn_hub2.py
#srun -p cp100 python mdn_hub_sdss.py
#srun -p cp100 python mdn_hub_cosmos.py
#srun -p cp100 python mdn_mag_sdss.py
#srun -p cp100 python mdn_mag_cosmos.py
srun -p cp100 python mdn_4_tf2.py

echo [$SECONDS] End job 
