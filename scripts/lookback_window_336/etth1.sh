python train.py --config_file config/gridtst/etth1.yaml --label_len 96 
python train.py --config_file config/gridtst/etth1.yaml --label_len 192  
python train.py --config_file config/gridtst/etth1.yaml --label_len 336 --attention_dropout 0.1 --d_model 128 --patch_len 48 --stride 96
python train.py --config_file config/gridtst/etth1.yaml --label_len 720 --attention_dropout 0.0 --attention_strategy alternate --d_model 256 --patch_len 64 --stride 16
