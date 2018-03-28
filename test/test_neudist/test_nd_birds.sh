
name='neural_dist_birds'
testing_path='../../Data/Results/birds/Finaltesting_num_10/birds_256_G_epoch_500.h5'
CUDA_VISIBLE_DEVICES=${device} python test_nd_worker.py --dataset birds --testing_path ${testing_path} --model_name ${name} --load_from_epoch 110


