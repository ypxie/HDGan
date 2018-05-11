
# evaluate birds
name='birds_256'
testing_path='../Results/birds/birds_512_testing_num_10/birds_512_G_epoch_80.h5'
CUDA_VISIBLE_DEVICES=${device} python neudist/neudist.py \
                        --dataset birds \
                        --testing_path ${testing_path} \
                        --model_name ${name} \
                        --load_from_epoch 110

# evaluate flowers
# name='neural_dist_flowers'
# testing_path='../../Data/Results/flowers/Finaltesting_num_26/flower_256_G_epoch_580.h5'
# CUDA_VISIBLE_DEVICES=${device} python test_nd_worker.py 
#                         --dataset flowers \
#                         --testing_path ${testing_path} \
#                         --model_name ${name} \
#                         --load_from_epoch 110


