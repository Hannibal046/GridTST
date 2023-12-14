for label_len in 96 192 336 720
do
    accelerate launch --num_processes 4 --gpu_ids 0,1,2,3 \
        train.py \
        --config_file config/gridtst/traffic.yaml \
        --seq_len 336 --label_len ${label_len} \
        --variate_sample_ratio 0.4 
done