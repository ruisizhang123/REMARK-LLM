mode=$1
if [[ "$mode" == "train" ]]; then
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --model_path t5-base\
        --datafile_path "Hello-SimpleAI/HC3-gpt" \
        --num_epochs 300 \
        --per_device_batch_size 16\
        --input_max_length 80\
        --target_max_length 80\
        --lr 3e-5\
        --save_path ./logs_text_wm\
        --gradient_accumulation_steps 8\
        --train_subset 0\
        --train_semantic 1\
        --train_attack 1\
        --train_semantic_loss cross_entrophy\
        --message_max_length 16\
        --wm_embed_model t5\
        --debug 1\
        --visualize 1\
        --target_text_type original\
        --wm 1\
        --inference_strategy token_idx\
        --inference_batch 1\
        --mapper_info logits\
        --train_rephrase 1\
        --attack 0\
        --schedule_tmp 0\
        --figurepint 0\
        --periodical 1\
        --discriminator 1\
        --adaptive 0\
        --message_embed_method addition_same
elif  [[ "$mode" == "val" ]]; then
    CUDA_VISIBLE_DEVICES=0 python inference.py \
        --load_path  ./logs_text_wm/debug \
        --dataset "NicolaiSivesind/ChatGPT-Research-Abstracts"\
        --save_path  ./logs_text_wm/debug \
        --model_path t5-base\
        --input_max_length 80\
        --target_max_length 80\
        --mask_per 0.3\
        --beam_width 5\
        --repeat 10\
        --message_max_length 16
fi 