#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=psuedo
#SBATCH --mail-type=END
#SBATCH --mail-user=ks4883@nyu.edu
#SBATCH --output=slurm_%j.out

date
module purge
module load python3/intel/3.6.3
source ~/pyenv/py3.6.3/bin/activate
cd /scratch/ks4883/pseudo
python plabel.py

date
