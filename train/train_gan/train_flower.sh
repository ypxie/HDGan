
name='train_flowers_256'

CUDA_VISIBLE_DEVICES="0" python train_worker.py --dataset flowers --batch_size 16 --model_name ${name}  --g_lr 0.0002 --d_lr 0.0002 


