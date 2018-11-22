
data_dir='/home/shigaki/Data'
name='sketch256_train'
dataset='coco'
dir='../../Models/'${name}_$dataset
mkdir -v $dir
CUDA_VISIBLE_DEVICES='0' python train_worker.py \
                                --data_dir $data_dir
                                --dataset $dataset \
                                --batch_size 8 \
                                --model_name ${name} \
                                --g_lr 0.0002 \
                                --d_lr 0.0002 \
                                --epoch_decay 50 \
                                --KL_COE 2 \
                                --gpus '0' \
                                | tee $dir/'log.txt'

# need about 150 epochs
