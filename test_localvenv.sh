#!/bin/bash
#SBATCH --nodes 1 
#SBATCH --gpus-per-node=1
#SBATCH --mem=31200M
#SBATCH --time=48:00:00
#SBATCH --account=rrg-bengioy-ad

cd /home/shuyuan/shuyuan/HRAC

source ../p311/bin/activate

python main.py --env_name AntMaze
