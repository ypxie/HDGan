
name='neural_dist'

CUDA_VISIBLE_DEVICES=${device} python train_nd_worker.py --dataset flowers --batch_size 32 --model_name ${name} 


