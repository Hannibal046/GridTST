for label_len in 96 192 336 720
do
    python train.py --config_file config/gridtst/solar.yaml --label_len ${label_len}
done