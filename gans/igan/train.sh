#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=15:00:00
#SBATCH --mem=30GB
#SBATCH --gres=gpu:k80:1
#SBATCH --job-name=I_GAN_FC_60
#SBATCH --mail-type=END
#SBATCH --mail-user=ks4883@nyu.edu
#SBATCH --output=slurm_%j.out

date
module purge
module load python3/intel/3.6.3
source ~/pyenv/py3.6.3/bin/activate
cd /scratch/ks4883/ssl_gan
python ImprovedGAN.py --cuda 

date
