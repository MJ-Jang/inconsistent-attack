DATA_DIR=$1
SAVE_DIR=$2

# You may change this part as variables
DATASET=esnli
MODEL_DIR=../model_binary/

cmd="CUDA_VISIBLE_DEVICES=0  python ./training/inference.py  \
    --dataset=$DATASET
    --data_directory=$DATA_DIR \
    --save_dir=$SAVE_DIR  \
    --model_directory=$MODEL_DIR"
echo $cmd
eval $cmd
