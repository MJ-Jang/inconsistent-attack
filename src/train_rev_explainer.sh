DATASET=$1
DATA_DIR=$2
MODEL_SAVE_DIR=$3

DIR='pwd'

python ./training/train.py --dataset $DATASET --data_directory $DATA_DIR --save_dir $MODEL_SAVE_DIR