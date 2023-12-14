for label_len in 96 192 336 720
do
    python train.py --config_file config/gridtst/electricity.yaml \
                    --seq_len 192 \
                    --label_len ${label_len} \
                    --patch_len 64 --stride 32
done
