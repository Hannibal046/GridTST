for label_len in 96 192 336 720
do
    accelerate launch --num_processes 4 --gpu_ids 0,1,2,3 \
        --main_process_port 29600 train.py --config_file config/gridtst/traffic.yaml --seq_len 512 --label_len ${label_len} \
        --patch_len 128 --stride 128 --variate_sample_ratio 0.4
done

