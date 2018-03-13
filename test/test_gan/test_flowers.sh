
name='flower_256'
CUDA_VISIBLE_DEVICES=${device} python test_worker.py --dataset flowers --model_name ${name} --load_from_epoch 580 --test_sample_num 26
 

