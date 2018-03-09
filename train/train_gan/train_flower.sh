
name='train_flowers_256'

CUDA_VISIBLE_DEVICES=${device} python train_worker.py --dataset flowers --batch_size 16 --model_name ${name} --reuse_weights --load_from_epoch 0 --g_lr 0.0002 --d_lr 0.0002 


