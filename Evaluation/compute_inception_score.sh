
# download /inception_finetuned_models in to inception_score/. See Readme

CUDA_VISIBLE_DEVICES=1 python inception_score/inception_score.py \
            --checkpoint_dir ./inception_score/inception_finetuned_models/birds_valid299/model.ckpt \
            --image_folder ../Results/birds/birds_256_testing_num_2 \
            --h5_file 'birds_256_G_epoch_500.h5' \
            --batch_size 32
    