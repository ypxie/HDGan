
name='HDGAN_256'
dataset='coco'
gpus=${gpus}
dir='../../Models/'${name}_$dataset
mkdir -v $dir
CUDA_VISIBLE_DEVICES=${device} python train_worker.py \
                                --dataset $dataset \
                                --batch_size 4 \
                                --model_name ${name} \
                                --g_lr 0.0002 \
                                --d_lr 0.0002 \
                                --epoch_decay 50 \
                                --KL_COE 2 \
                                --gpus ${device} \
                                | tee $dir/'log.txt'

