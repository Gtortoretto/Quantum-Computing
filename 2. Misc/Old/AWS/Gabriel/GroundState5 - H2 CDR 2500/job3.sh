#!/bin/bash

#SBATCH --job-name=teste       # Job name
#SBATCH --ntasks=1                     # Number of tasks, 1 for a simple script
#SBATCH --cpus-per-task=1              # Number of cores per task, assuming single-threaded
#SBATCH --nodes=1                      # Number of nodes, only 1 needed
#SBATCH --mem-per-cpu=600mb            # Adjust memory as needed for your script
#SBATCH --time=24:00:00                # Adjust wall time as needed
#SBATCH --output=log/job_%j.out
#SBATCH --error=log/job_%j.err

# module load python/3.12.4

source /home/gabriel/miniconda3/etc/profile.d/conda.sh

conda activate aws_braket

python task3.py