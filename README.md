# Architecture of Dynamic Spatial-temporal Hypergraph Convolutional Network for Skeleton-based Action Recognition

## Prerequisites

- Python = 3.8.8
- PyTorch = 1.10.0
- Run `pip install -e torchlight` 


#### There are 3 datasets to download:

- NTU RGB+D 60 Skeleton
- NTU RGB+D 120 Skeleton
- NW-UCLA

## NTU RGB+D 60 and 120 and NW-UCLA

1.Download the raw data from the website and place it in the appropriate directory of the './data' file
2.Generate NTU RGB+D 60 and NTU RGB+D 120 dataset:  python get_raw_skes_data.py,  python get_raw_denoised_data.py,  python seq_transformation.py
3. Place the processed data file into the data_path parameter inside the './config'

## Training & Testing

### Training

#### Example: training  DST-HCN on NTU RGB+D 120 cross subject, the training setup parameters for the other datasets are set under the './config' file 
python mainfucos.py --config config/nturgbd120-cross-set/default.yaml --work-dir "/mnt/data/demo" --device 1 2 --num-epoch 90

### Testing

- To test the trained models saved in <work_dir>:
python mainfucos.py --config <work_dir>/config.yaml --work-dir <work_dir>  --weights <work_dir>/.pt 


- To ensemble the results of different streams
python zhenghe.py 

### Pretrained Models

We provide individual stream weighting files for the relevant dataset

# Citation
## NOTE：Our work and acceptance by ICME 2023 Oral
Please cite this work if you find it useful:


@INPROCEEDINGS{10220028,
  author={Wang, Shengqin and Zhang, Yongji and Qi, Hong and Zhao, Minghao and Jiang, Yu},
  booktitle={2023 IEEE International Conference on Multimedia and Expo (ICME)}, 
  title={Dynamic Spatial-temporal Hypergraph Convolutional Network for Skeleton-based Action Recognition}, 
  year={2023},
  volume={},
  number={},
  pages={2147-2152},
  doi={10.1109/ICME55011.2023.00367}}


