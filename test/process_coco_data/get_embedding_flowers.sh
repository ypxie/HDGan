CUB_ENCODER='lm_sje_flowers_c10_hybrid_0.00070_1_10_trainvalids.txt_iter16400.t7' 
CAPTION_PATH='interp_exp/imagenet_cls_for_rebuttal/flowers/text_left' 

export CUDA_VISIBLE_DEVICES=2

net_txt=${CUB_ENCODER} \
queries=${CAPTION_PATH}.txt \
filenames=${CAPTION_PATH}.t7 \
th get_embedding.lua

python t7_to_pickle.py --t7_path ${CAPTION_PATH}.t7