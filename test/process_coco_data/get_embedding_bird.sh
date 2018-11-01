CUB_ENCODER='lm_sje_nc4_cub_hybrid_gru18_a1_c512_0.00070_1_10_trainvalids.txt_iter30000.t7' 
CAPTION_PATH='interp_exp/demo/birds/text_perword' # change this to target file

export CUDA_VISIBLE_DEVICES=2

net_txt=${CUB_ENCODER} \
queries=${CAPTION_PATH}.txt \
filenames=${CAPTION_PATH}.t7 \
th get_embedding.lua

# python t7_to_pickl    e.py --t7_path ${CAPTION_PATH}'.t7'