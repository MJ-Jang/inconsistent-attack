DATA_DIR=$1
MODEL_SAVE_DIR=$2

DIR='pwd'

python ./training/train.py --data_directory $DATA_DIR --save_dir $MODEL_SAVE_DIR
