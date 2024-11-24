# Multi-Scale Adaptive Convolutional Graph for Multi-Stream Concept Drift

This is a PyTorch implementation of the paper "[Multi-Scale Adaptive Convolutional Graph for Multi-Stream Concept Drift]"

## Installation

Install the dependency using the following command:

```bash
pip install -r requirements.txt
```

* torch
* scipy>=0.19.0
* numpy>=1.12.1
* pandas>=0.19.2
* pyyaml==5.3.1
* statsmodels
* tensorflow>=1.3.0
* tables
* future


## Data Preparation

The traffic data files for Los Angeles (METR-LA) and the Bay Area (PEMS-BAY) are put into the `data/` folder. They are provided by [DCRNN](https://github.com/chnsh/DCRNN_PyTorch).

Run the following commands to generate train/test/val dataset at  `data/Processed Data/{train,val,test}.npz`.
```bash

# METR-LA
python -m scripts.Split_data --output_dir=data/Processed Data --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python -m scripts.Split_data --output_dir=data/Processed Data --traffic_df_filename=data/pems-bay.h5

# feature1 (Weather)
python -m scripts.Split_data_fg --output_dir=data/Processed Data --traffic_df_filename=data/FeatureGroup/feature1.csv

# feature3 (Weather)
python -m scripts.Split_data_fg --output_dir=data/Processed Data --traffic_df_filename=data/FeatureGroup/feature3.csv

# feature6 (Weather)
python -m scripts.Split_data_fg --output_dir=data/Processed Data --traffic_df_filename=data/FeatureGroup/feature6.csv

```

## Train Model

When you train the model, you can run:

```bash
# Use METR-LA dataset
python train.py --config_filename=data/Para/para_la.yaml --temperature=0.9

# Use PEMS-BAY dataset
python train.py --config_filename=data/Para/para_bay.yaml --temperature=0.9

# Use Weather dataset
python train.py --config_filename=data/Para/para_fg.yaml --temperature=0.9

```

Hyperparameters can be modified in the `para_la.yaml`, `para_bay.yaml` and `para_fg.yaml` files.


## Citation

If you use this repository, e.g., the code and the datasets, in your research, please cite the following paper:
```

```

## Acknowledgments
[DCRNN-PyTorch](https://github.com/chnsh/DCRNN_PyTorch), [GTS](https://github.com/chaoshangcs/GTS)
