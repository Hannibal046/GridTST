for label_len in 96 192 336 720
do
    python train.py --config_file config/gridtst/electricity.yaml \
                    --seq_len 96 \
                    --label_len ${label_len} \
                    --patch_len 8 --stride 8 \
                    --per_device_train_batch_size 16 --gradient_accumulation_steps 4
done
