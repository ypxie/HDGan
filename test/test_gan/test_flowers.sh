
name='flower_256'
CUDA_VISIBLE_DEVICES="0" python test_worker.py \
                                    --dataset birds \
                                    --model_name ${name} \
                                    --load_from_epoch 580 \
                                    --test_sample_num 26 \
                                    --finest_size 256 \
                                    --batch_size 16
