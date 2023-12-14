for label_len in 96 192 336 720
do
    accelerate launch --num_processes 4 --gpu_ids 0,1,2,3 \
        --main_process_port 29500 train_partial.py \
        --config_file config/gridtst/traffic.yaml \
        --seq_len 96 --label_len ${label_len} \
        --patch_len 16 --stride 8 \
        --per_device_train_batch_size 6 --gradient_accumulation_steps 2 \
        --variate_sample_ratio 0.4 
done