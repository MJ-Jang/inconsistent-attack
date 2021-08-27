GPUDEV=$1
DATA_DIR=$2
MODEL_SAVE_DIR=$3

# You may change this part as variables
DATASET=esnli
EPOCHS=5
LOGGING_STEPS=100
#LOGGING_STEPS=10000
BATCH_SIZE=32


cmd="CUDA_VISIBLE_DEVICES="$GPUDEV"  python ./training/train.py  \
    --dataset=$DATASET
    --data_directory=$DATA_DIR \
    --save_dir=$MODEL_SAVE_DIR  \
    --per_gpu_train_batch_size $BATCH_SIZE \
    --per_gpu_eval_batch_size $BATCH_SIZE \
    --logging_steps $LOGGING_STEPS \
    --num_train_epochs $EPOCHS"
echo $cmd
eval $cmd
