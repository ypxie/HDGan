
name='HDGAN_256_v2_coco'
CUDA_VISIBLE_DEVICES=${device} python test_worker.py \
                                --dataset coco \
                                --model_name ${name} \
                                --load_from_epoch 50 \
                                --test_sample_num 1 \
                                --batch_size 8 \

