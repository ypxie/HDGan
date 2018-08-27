## test 512 resolution
name='birds_512'
CUDA_VISIBLE_DEVICES="0" python test_worker.py \
                                    --dataset birds \
                                    --model_name ${name} \
                                    --load_from_epoch 80 \
                                    --test_sample_num 10 \
                                    --finest_size 512 \
                                    --batch_size 4

# test 256
# name='HDGAN_256_birds'
# CUDA_VISIBLE_DEVICES="0" python test_worker.py \
#                                     --dataset birds \
#                                     --model_name ${name} \
#                                     --load_from_epoch 500 \
#                                     --test_sample_num 10 \
#                                     --save_visual_results \
#                                     --batch_size 8


