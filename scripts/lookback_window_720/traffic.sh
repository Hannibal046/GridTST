for label_len in 96 192 336 720
do
    accelerate launch --num_processes 4 --gpu_ids 4,5,6,7 \
        --main_process_port 29500 train.py --config_file config/gridtst/traffic.yaml --seq_len 720 --label_len ${label_len} \
        --patch_len 196 --stride 196 --variate_sample_ratio 0.4
done