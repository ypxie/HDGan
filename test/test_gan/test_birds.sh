
name='birds_256'
CUDA_VISIBLE_DEVICES=${device} python test_worker.py --dataset birds --model_name ${name} --load_from_epoch 500 --test_sample_num 2 --batch_size 32
 

