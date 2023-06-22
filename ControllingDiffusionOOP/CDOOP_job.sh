#!/usr/bin/env bash
#SBATCH -A SNIC2022-5-398 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -t 0-05:30:00

apptainer build ENV_image.sif ENV_recipe.def

apptainer build --force CDOOP_image.sif CDOOP_recipe.def

apptainer exec --nv CDOOP_image.sif python -m unittest discover -s /job_scripts

apptainer exec --nv CDOOP_image.sif python job_scripts/main.py