#!/bin/sh
#SBATCH -p short
#SBATCH -n1

python3 ./generate_data.py --street $1 --start-idx $2
