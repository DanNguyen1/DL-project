#!/usr/bin/bash
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH --ntasks=4                   # Number of CPU cores
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --mem=16gb                   # Job memory request
#SBATCH --time=03:50:00              # Time limit hrs:min:sec
#SBATCH --qos short
#SBATCH -o slurm-logs/%j-stdout.txt
#SBATCH -e slurm-logs/%j-stderr.txt
#SBATCH --mail-type=begin       # send email when job begins
#SBATCH --mail-type=end         # send email when job ends
#SBATCH --mail-user=andrea_pierre@student.uml.edu
#SBATCH --mail-type=TIME_LIMIT_80
#
#
## Load module
module load cuda/12.6
module load cudnn/8.9.7.29-12-cuda12.6
## Load venv env
source .venv/bin/activate
pip install -Ue .
## Run script
python training.py
