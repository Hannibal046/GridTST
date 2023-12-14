python train.py --config_file config/gridtst/ettm1.yaml --label_len 96
python train.py --config_file config/gridtst/ettm1.yaml --label_len 192
python train.py --config_file config/gridtst/ettm1.yaml --label_len 336 --d_model 32 --patch_len 32 --stride 16
python train.py --config_file config/gridtst/ettm1.yaml --label_len 720 --num_layer 5 --stride 4 --patch_len 4
