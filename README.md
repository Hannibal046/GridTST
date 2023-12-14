# GridTST
> A Time Series is Worth 16x16 Words: Long-term Forecasting with GridTST

|                          | vanilla Transformer | Multivariate Modeling | Sequntial Modeling |
|:------------------------:|:-------------------:|:---------------------:|:------------------:|
|    DLinear (AAAI2023)    |          ❌          |           ❌           |          ❌         |
|  CrossFormer (ICLR2023)  |          ❌          |           ✔️           |          ✔️         |
|    PatchTST (ICLR2023)   |          ✔️          |           ❌           |          ✔️         |
| iTransformer (ICLR 2024) |          ✔️          |           ✔️           |          ❌         |
|        **GridTST**       |          ✔️          |           ✔️           |          ✔️         |


|    Model    |   GridTST   | [PatchTST](https://openreview.net/forum?id=Jbdc0vTOcol)  (ICLR 2023) | [iTransformer](https://arxiv.org/abs/2310.06625)  (ICLR 2024)  | [Dlinear](https://arxiv.org/abs/2205.13504)  (AAAI 2023) |
|:-----------:|:-----------:|:-------------------:|:------------------------:|:------------------:|
|   Weather   | **0.223** |              0.228 |                  0.236 |              0.246 |
|   Traffic   | **0.372** |              0.396 |                    0.386 |            0.433 |
| Electricity | **0.152** |              0.163 |                  0.165 |            0.166 |
|   Illness   | **1.649** |             1.806 |                  2.122|              2.169 |
|    Etth1    |   **0.416** |             0.421 |                     0.450 |            0.422 |
|    Ettm1    | **0.345** |               0.351 |                   0.365 |              0.357 |
|    Solar    | **0.187** |              0.215 |                  0.215 |            0.244 |

## Requirements
We recommand to use Conda to mange a virtual environment:
```bash
conda create -n gridtst python=3.8 && conda activate gridtst
pip install -r requirements.txt
```
logging and multi-gpu training setup:
```bash
wandb login
accelerate config
```

## Datasets
This is the dataset we use, you could download [here](https://drive.google.com/drive/folders/16DqgnUZEXd6Vmth-tL9e5JVnadh90GwF?usp=sharing) and put all csv files in the `dataset` folder.
|    Datast   | # Channels | # TimeSteps | Prediction Length |   Information  |
|:-----------:|:----------:|:-----------:|:-----------------:|:--------------:|
|   Weather   |     21     |    52696    |  {96,192,336,720} |     Weather    |
|   Traffic   |     862    |    17544    |  {96,192,336,720} | Transportation |
| Electricity |     321    |    26304    |  {96,192,336,720} |   Electricity  |
|   Illness   |      7     |     966     |   {12,24,48,60}   |     Illness    |
|    Etth1    |      7     |    17420    |  {96,192,336,720} |   Electricity  |
|    Ettm1    |      7     |    69680    |  {96,192,336,720} |   Electricity  |
|    Solar    |     137    |    52560    |  {96,192,336,720} |     Energy     |


## Get Started
We provide all the scripts on the `scripts` folder.
For example, training on the `Weather` dataset with `lookback window = 336`:
```bash
bash scripts/lookback_window_336/weather.sh
```

## Available Checkpoints
We provide our trained model on the [huggingface space](https://huggingface.co/GridTST). 

To evaluate these models, you could either specify a perticular model or evaluate them all at once.

For a certain model, for example `GridTST` on `traffic` dataset with `lookback window=336` and `prediction length=96`:
```bash
python benchmark.py --data_file dataset/traffic.csv --seq_len 336 --label_len 96
```

To evaluate them all:
```bash
python benchmark.py --all
```