
# download /inception_finetuned_models in to inception_score/. See Readme

# CUDA_VISIBLE_DEVICES=${device} python inception_score/inception_score.py \
#             --checkpoint_dir ./inception_score/inception_finetuned_models/birds_valid299/model.ckpt \
#             --image_folder ../Results/birds/birds_512_testing_num_10 \
#             --h5_file 'birds_512_G_epoch_80.h5' \
#             --batch_size 32

# coco
CUDA_VISIBLE_DEVICES=${device} python inception_score/inception_score_coco.py \
            --image_folder ../Results/coco/coco_256_testing_num_1 \
            --h5_file 'coco_256_G_epoch_200.h5' \
            --batch_size 32