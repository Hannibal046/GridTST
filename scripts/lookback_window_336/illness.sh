python train.py \
    --config_file config/gridtst/illness.yaml \
    --label_len 24 \
    --stride 8 \
    --patch_len 64 \
    --lr 0.0002 \
    --pct_start 0.4

python train.py \
    --config_file config/gridtst/illness.yaml \
    --label_len 36 \
    --stride 4 \
    --patch_len 48 \
    --lr 0.00015 \
    --pct_start 0.4

python train.py \
    --config_file config/gridtst/illness.yaml \
    --label_len 48 \
    --stride 4 \
    --patch_len 48 \
    --lr 0.001 \
    --pct_start 0.3

python train.py \
    --config_file config/gridtst/illness.yaml \
    --label_len 60 \
    --d_model 96 \
    --lr 0.00015 \
    --patch_len 16 \
    --dropout 0.2 \
    --attention_dropout 0.1 \
    --norm_type batchnorm \
    --patch_len 48 \
    --stride 8 \
    --num_layers 5
