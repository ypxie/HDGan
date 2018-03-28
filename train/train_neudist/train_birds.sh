
name='neural_dist'

CUDA_VISIBLE_DEVICES="0" python train_nd_worker.py --dataset birds --batch_size 32 --model_name ${name} 