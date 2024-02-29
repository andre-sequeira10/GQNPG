#!/bin/bash -l
#
#SBATCH --job-name=qfim_block_diag
#
#SBATCH --ntasks-per-node=10
#SBATCH --nodes=2
#SBATCH --time=5:00:00

module load Python/3.10.4-GCCcore-11.3.0

srun python qnpg_reinforce_cartpole.py --init zeros --policy Q --ng 1 --n_layers 4 --episodes 500 --entanglement all2all --batch_size 3 --filename_save "qfim_block_diag"
