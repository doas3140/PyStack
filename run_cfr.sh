#!/bin/sh
#SBATCH -p short
#SBATCH -n1

python3 ./DataGeneration/main_data_generation.py --street $1 --start-idx $2
