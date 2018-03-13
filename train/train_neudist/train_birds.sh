
name='neural_dist'

CUDA_VISIBLE_DEVICES=${device} python train_nd_worker.py --dataset birds --batch_size 32 --model_name ${name} --reuse_weights --load_from_epoch 0


