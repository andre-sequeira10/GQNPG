#!/bin/bash -l
#
#SBATCH --job-name=fim_half
#
#SBATCH --ntasks-per-node=10
#SBATCH --nodes=3
#SBATCH --time=6:00:00

module load Python/3.10.4-GCCcore-11.3.0

srun python qnpg_reinforce_acrobot_born.py --init zeros --policy Q --ng 1 --n_layers 4 --episodes 500 --entanglement all2all --batch_size 3 --filename_save "fim_half"
