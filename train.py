## built-in
import math,logging,os
import types
os.environ["TOKENIZERS_PARALLELISM"]='true'

## third-party
from accelerate import Accelerator
from accelerate.logging import get_logger
import transformers
transformers.logging.set_verbosity_error()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

## own
from utils import (
    get_yaml_file,
    set_seed,
    MAE,
    MSE,
)
from model import (
    GridTSTConfig,
    GridTSTForTimeSeriesPrediction,
)
torch.autograd.set_detect_anomaly(True)
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    ## adding args here for more control from CLI is possible
    parser.add_argument("--config_file",default='config/train_patchtst_traffic.yaml')

    parser.add_argument("--per_device_train_batch_size",type=int)
    parser.add_argument("--per_device_eval_batch_size",type=int)
    parser.add_argument("--gradient_accumulation_steps",type=int)
    parser.add_argument("--lr",type=float)
    parser.add_argument("--pct_start",type=float)
    parser.add_argument("--weight_decay",type=float)
    parser.add_argument("--max_grad_norm",type=float)
    parser.add_argument("--fp16",type=eval)
    parser.add_argument("--torch_compile",type=eval)


    parser.add_argument("--pooling_head")
    parser.add_argument("--seq_len",type=int)
    parser.add_argument("--label_len",type=int)
    parser.add_argument("--stride",type=int)
    parser.add_argument("--patch_len",type=int)
    parser.add_argument("--max_train_epochs",type=int)
    parser.add_argument("--num_patience",type=int,default=20)
    
    parser.add_argument("--d_model",type=int)
    parser.add_argument("--ffn_dim",type=int)
    parser.add_argument("--num_layers",type=int)
    parser.add_argument("--dropout",type=float)
    parser.add_argument("--attention_dropout",type=float)
    parser.add_argument("--revin_affine",type=eval)
    parser.add_argument("--norm_type")
    parser.add_argument("--attention_strategy")
    parser.add_argument("--variate_sample_ratio",type=float,default=1.0)

    parser.add_argument("--group_name")
    parser.add_argument("--project_name",default='GridTST')


    args = parser.parse_args()

    yaml_config = get_yaml_file(args.config_file)
    args_dict = {k:v for k,v in vars(args).items() if v is not None}
    yaml_config.update(args_dict)
    args = types.SimpleNamespace(**yaml_config)
    return args

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self,file_path,seq_len,label_len,stage,variate_sample_ratio=1.0):
        self.seq_len = seq_len
        self.label_len = label_len
        self.stage = stage
        self.scaler = StandardScaler()
        self.variate_sample_ratio = variate_sample_ratio

        df = pd.read_csv(file_path)
        
        if "ETTh" in file_path:
            borders = {
                "train": [0,12 * 30 * 24],
                "dev": [12 * 30 * 24 - self.seq_len,12 * 30 * 24 + 4 * 30 * 24],
                "test":[12 * 30 * 24 + 4 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 8 * 30 * 24]
            }
        elif "ETTm" in file_path:
            borders = {
                "train": [0,12 * 30 * 24 * 4],
                "dev": [12 * 30 * 24 * 4 - self.seq_len,12 * 30 * 24 * 4 + 4 * 30 * 24 * 4],
                "test":[12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
            }

        else:
            num_train = int(len(df)*0.7)
            num_test = int(len(df)*0.2)
            num_dev = len(df) - num_train - num_test

            borders = {
                "train": [0,                             num_train],
                "dev":   [num_train-self.seq_len,        num_train+num_dev],
                "test":  [len(df)-num_test-self.seq_len, len(df)]
            }

        df = df[df.columns[1:]]
        train_data = df[borders['train'][0]:borders['train'][1]]
        self.scaler.fit(train_data.values)
        df = self.scaler.transform(df.values)

        self.data = df[borders[stage][0]:borders[stage][1]]
        self.num_channels = self.data[0].shape[0]

    def __len__(self):
       return len(self.data) - self.seq_len - self.label_len + 1
    
    def __getitem__(self,idx):
        if self.stage != 'train' or self.variate_sample_ratio == 1.0:
            return (self.data[idx             :  idx+self.seq_len],
                    self.data[idx+self.seq_len:  idx+self.seq_len+self.label_len])
        else:
            num_variates_to_sample = int(self.num_channels * self.variate_sample_ratio)
            sampled_indices = np.random.choice(self.num_channels, num_variates_to_sample, replace=False)
            return (self.data[idx             :  idx+self.seq_len][:,sampled_indices],
                    self.data[idx+self.seq_len:  idx+self.seq_len+self.label_len][:,sampled_indices])

def validate(model,dataloader,accelerator):
    model.eval()
    preds,labels = [],[]
    for (inputs,label) in dataloader:
        with torch.no_grad():
            pred = model(inputs.float()).detach().cpu().numpy()
        label = label.float().cpu().numpy()
        preds.append(pred)
        labels.append(label)
            
    if accelerator.use_distributed and accelerator.num_processes>1:
        num_batch_in_one_process = len(preds)
        preds_from_all_gpus = [None for _ in range(accelerator.num_processes)] 
        dist.all_gather_object(preds_from_all_gpus,preds)
        preds = [
                preds_from_all_gpus[rank_idx][batch_idx] 
                    for batch_idx in range(num_batch_in_one_process) 
                        for rank_idx in range(accelerator.num_processes)
                ]

        labels_from_all_gpus = [None for _ in range(accelerator.num_processes)] 
        dist.all_gather_object(labels_from_all_gpus,labels)
        labels = [
                    labels_from_all_gpus[rank_idx][batch_idx] 
                        for batch_idx in range(num_batch_in_one_process) 
                            for rank_idx in range(accelerator.num_processes)
                ]
    
    preds  = np.concatenate(preds,axis=0)[:len(dataloader.dataset)]
    labels = np.concatenate(labels,axis=0)[:len(dataloader.dataset)]
    
    return MSE(preds,labels),MAE(preds,labels)

def main(args):
    set_seed(args.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with='wandb',
        mixed_precision='no',
    )

    accelerator.init_trackers(
        project_name=args.project_name,
        config=args,
        init_kwargs={"wandb": {"group": args.group_name if hasattr(args,"group_name") else None,}},
    )

    if accelerator.is_local_main_process:
        wandb_tracker = accelerator.get_tracker("wandb")
        LOG_DIR = wandb_tracker.run.dir

    if args.model == 'gridtst':
        config = GridTSTConfig(
            num_channels=args.num_channels,seq_len=args.seq_len,label_len=args.label_len,stride=args.stride,patch_len=args.patch_len,
            num_layers = args.num_layers,d_model=args.d_model,revin_affine=args.revin_affine,norm_type=args.norm_type,attention_strategy=args.attention_strategy,attention_dropout=args.attention_dropout,
            ffn_dim=args.ffn_dim,dropout=args.dropout,
            )
        model  = GridTSTForTimeSeriesPrediction(config)
        
    model.train()

    train_dataset = TimeSeriesDataset(args.data_file,args.seq_len,args.label_len,'train',args.variate_sample_ratio)
    dev_dataset   = TimeSeriesDataset(args.data_file,args.seq_len,args.label_len,'dev')
    test_dataset  = TimeSeriesDataset(args.data_file,args.seq_len,args.label_len,'test')

    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.per_device_train_batch_size,shuffle=True,drop_last=False,num_workers=4,pin_memory=True)
    dev_dataloader   = torch.utils.data.DataLoader(dev_dataset,batch_size=args.per_device_eval_batch_size,shuffle=False,drop_last=False,num_workers=4,pin_memory=True)
    test_dataloader  = torch.utils.data.DataLoader(test_dataset,batch_size=args.per_device_eval_batch_size,shuffle=False,drop_last=False,num_workers=4,pin_memory=True)


    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,lr=args.lr)
    
    model, optimizer, train_dataloader, dev_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, dev_dataloader,test_dataloader
    )
    
    BEST_DEV_MSE=100
    BEST_TEST_MSE=100
    BEST_TEST_MAE=100
    PATIENCE = args.num_patience
    SHOULD_BREAK=False
    NUM_UPDATES_PER_EPOCH = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    MAX_TRAIN_STEPS = NUM_UPDATES_PER_EPOCH * args.max_train_epochs
    MAX_TRAIN_EPOCHS = math.ceil(MAX_TRAIN_STEPS / NUM_UPDATES_PER_EPOCH)
    TOTAL_TRAIN_BATCH_SIZE = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    EVAL_STEPS = args.val_check_interval if isinstance(args.val_check_interval,int) else int(args.val_check_interval * NUM_UPDATES_PER_EPOCH)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer = optimizer,
                                                steps_per_epoch = NUM_UPDATES_PER_EPOCH,
                                                pct_start = args.pct_start,
                                                epochs = MAX_TRAIN_EPOCHS,
                                                max_lr = args.lr)
    
    progress_bar_postfix_dict = {}
    
    logger.info("***** Running training *****")
    logger.info(f"  Dataset = {args.data_file}")
    logger.info(f"  Num Channels = {args.num_channels}")
    if hasattr(args,'patch_len'):logger.info(f"  Patch Length = {args.patch_len}")
    if hasattr(args,'stride'):logger.info(f"  Stride = {args.stride}")
    logger.info(f"  Num Patches = {config.num_patches}")
    logger.info(f"  Sequence Length = {args.seq_len}")
    logger.info(f"  Label Length = {args.label_len}")
    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num dev examples = {len(dev_dataset)}")
    logger.info(f"  Num test examples = {len(test_dataset)}")
    logger.info(f"  Num Epochs = {MAX_TRAIN_EPOCHS}")
    logger.info(f"  Per device train batch size = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {TOTAL_TRAIN_BATCH_SIZE}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {MAX_TRAIN_STEPS}")
    logger.info(f"  Per device eval batch size = {args.per_device_eval_batch_size}")
    logger.info(f"  Model Size = {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.6f} M")
    completed_steps = 0
    trained_samples = 0
    progress_bar = tqdm(range(MAX_TRAIN_STEPS), disable=not accelerator.is_local_main_process,ncols=125)

    for epoch in range(MAX_TRAIN_EPOCHS):
        progress_bar.set_description(f"epoch: {epoch+1}/{MAX_TRAIN_EPOCHS}")
        for _ ,(inputs,labels) in enumerate(train_dataloader):
            trained_samples += inputs.shape[0]
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    inputs = inputs.float()
                    labels = labels.float()
                    loss = F.mse_loss(model(inputs),labels)
                    
                accelerator.backward(loss)
                ## one optimization step
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    progress_bar_postfix_dict.update(dict(loss=f"{loss:.4f}",lr=f"{optimizer.param_groups[0]['lr']:6f}"))
                    progress_bar.set_postfix(progress_bar_postfix_dict)
                    completed_steps += 1
                    if hasattr(args,'max_grad_norm'): accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    if not accelerator.optimizer_step_was_skipped:lr_scheduler.step()
                    optimizer.zero_grad()
                    accelerator.log({"trained_samples": trained_samples}, step=completed_steps)
                    accelerator.log({"training_loss": loss}, step=completed_steps)
                    accelerator.log({"lr": optimizer.param_groups[0]['lr']}, step=completed_steps)
                    
                    if completed_steps % EVAL_STEPS == 0:
                        dev_mse,dev_mae   = validate(model,dev_dataloader,accelerator)
                        test_mse,test_mae = validate(model,test_dataloader,accelerator)
                        model.train()
                        accelerator.log({"epoch": epoch+1}, step=completed_steps)
                        accelerator.log({"dev_mse": dev_mse}, step=completed_steps)
                        if dev_mse < BEST_DEV_MSE:
                            PATIENCE = args.num_patience
                            BEST_DEV_MSE = dev_mse
                            BEST_TEST_MAE = test_mae
                            BEST_TEST_MSE = test_mse
                            accelerator.log({"test_mse":test_mse}, step=completed_steps)
                            progress_bar_postfix_dict.update(dict(test_mse=f"{test_mse:.4f}"))
                            accelerator.wait_for_everyone()
                            if accelerator.is_local_main_process:
                                unwrapped_model = accelerator.unwrap_model(model)
                                unwrapped_model.save_pretrained(os.path.join(LOG_DIR,"ckpt"))
                            accelerator.wait_for_everyone()
                        else:
                            PATIENCE -= 1
                            if PATIENCE <= 0:
                                SHOULD_BREAK = True
                                break
        
        if SHOULD_BREAK:break       

    accelerator.log({"final mse":BEST_TEST_MSE}, step=completed_steps)
    accelerator.log({"final mae":BEST_TEST_MAE}, step=completed_steps)
    if accelerator.is_local_main_process:
        print(f"\ntest mse:{BEST_TEST_MSE:.4f} test_mae:{BEST_TEST_MAE:.4f}")
    accelerator.end_training()

if __name__ == '__main__':
    args = parse_args()
    main(args)