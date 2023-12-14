from model import GridTSTForTimeSeriesPrediction
from train import TimeSeriesDataset
import argparse
import os
import torch
import numpy as np
import json
from utils import (
    MAE,
    MSE,
)
from tqdm import tqdm

datafile2name = {
    'weather.csv':'weather',
    'traffic.csv':"traffic",
    'electricity.csv':"electricity",
    "national_illness.csv":'illness',
    "ETTh1.csv":'etth1',
    "ETTm1.csv":'ettm1',
    'solar.csv':'solar',
}

def benchmark(data_file,seq_len,label_len,batch_size,):

    dataset = TimeSeriesDataset(data_file,seq_len,label_len,'test')
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=False,drop_last=False)

    pretrained_model_name = f"GridTST/{datafile2name[os.path.basename(data_file)]}_{seq_len}_{label_len}"
    model = GridTSTForTimeSeriesPrediction.from_pretrained(pretrained_model_name).cuda()
    model.eval()

    preds,labels = [],[]
    for (inputs,label) in tqdm(dataloader,desc=f'Testing {pretrained_model_name}...'):
        with torch.no_grad():
            label = label.cuda()
            inputs = inputs.cuda()
            pred = model(inputs.float()).detach().cpu().numpy()
        label = label.float().cpu().numpy()
        preds.append(pred)
        labels.append(label)
    
    preds  = np.concatenate(preds,axis=0)[:len(dataloader.dataset)]
    labels = np.concatenate(labels,axis=0)[:len(dataloader.dataset)]
    
    mse,mae = MSE(preds,labels),MAE(preds,labels)
    return mse.item(),mae.item(),len(dataloader.dataset)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file")
    parser.add_argument("--seq_len",type=int)
    parser.add_argument("--label_len",type=int)
    parser.add_argument("--batch_size",type=int,default=48)
    parser.add_argument("--all",action='store_true')
    args = parser.parse_args()
    
    if args.all:
        results = []
        for dataset in datafile2name.keys():
            label_lens = [96,192,336,720] if dataset != 'national_illness.csv' else [24,36,48,60]
            seq_len = 336 if dataset != 'national_illness.csv' else 104
            data_file = os.path.join("dataset",dataset)
            for label_len in label_lens:
                mse,mae,num_test_samples = benchmark(data_file,seq_len,label_len,args.batch_size)
                results.append(
                    {
                        "dataset": datafile2name[dataset],
                        "lookback window":seq_len,
                        "prediction length":label_len,
                        "# test samples":num_test_samples,
                        "MSE":f"{mse:.4f}",
                        "MAE":f"{mae:.4f}",
                    }
                )
        for result in results:
            print(json.dumps(result,indent=4))

    else:
        mse,mae,num_test_samples = benchmark(args.data_file,args.seq_len,args.label_len,args.batch_size)
        print(json.dumps(
                    {
                        "dataset": datafile2name[os.path.basename(args.data_file)],
                        "lookback window":args.seq_len,
                        "prediction length":args.label_len,
                        "# test samples":num_test_samples,
                        "MSE":f"{mse:.4f}",
                        "MAE":f"{mae:.4f}",
                    },
                    indent=4)
            )
        



