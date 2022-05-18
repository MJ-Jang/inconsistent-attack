#!/bin/bash

#SBATCH --job-name=mj-nile-expl

# set the number of nodes
#SBATCH --nodes=1
# set max wallclock time
#SBATCH --time=120:00:00
#SBATCH --partition=small
# how many gpus you want to use
#SBATCH --gres=gpu:1
#SBATCH --mail-user=myeongjun.jang@cs.ox.ac.uk

module purge
module load cuda/9.2
module load python3/anaconda
source /jmain02/home/J2AD009/ttl04/mmj86-ttl04/inconsistent-attack/venv/bin/activate
#/jmain02/home/J2AD009/ttl04/mmj86-ttl04/

HOME_DIR="/jmain02/home/J2AD009/ttl04/mmj86-ttl04"                  # ADJUST THIS TO YOUR HOME DIRECTORY
APP_DIR="${HOME_DIR}/inconsistent-attack"                          # ADJUST THIS TO YOUR PROJECT DIRECTORY
PYTHON="${HOME_DIR}/inconsistent-attack/venv/bin/python"                # ADJUST THIS TO YOUR CONDA ENV

export PYTHONPATH="/jmain02/home/J2AD009/ttl04/mmj86-ttl04/inconsistent-attack:${PYTHONPATH}"
export TRANSFORMERS_CACHE=$HOME_DIR/.cache/huggingface/transformers
export TRANSFORMERS_OFFLINE=1

#  CHANGING PARAMS  ############################################################################################
# BATCH_SIZE="16"
# ACCUM="4"
# SAVE_STEPS="3000"
# TTYPE="expl"
# IGNORE_IMG="1"
# FOLDER_NAME="/experiments/esnlive/me-noIMG"


#  MODEL / TRAINING CONFIG  ############################################################################################


cd /jmain02/home/J2AD009/ttl04/mmj86-ttl04/inconsistent-attack/src
# if they are run from somewhere else it will not work
#bash train_rev_explainer.sh 0 ../resources/cose_knowcage/ ../model_binary/
#python generate_inconsistent_expl.py --data_dir ../resources/esnli_knownile/ --save_dir ../resources/esnli_knownile/
bash generate_inconsistent_var.sh ../resources/cose_knowcage/ ../resources/cose_knowcage/


