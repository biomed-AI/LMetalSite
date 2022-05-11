# Introduction
MetalSite is an alignment-free sequence-based metal ion-binding site predictor based on pretrained language model and multi-task learning. MetalSite is easy to install and use, and is also accurate (surpassing the state-of-the-art structure-based methods) and really fast. Prediction of 500 sequences with an average length of 300 only takes about 2 minutes using GPU, or 20 minutes using CPU.
![MetalSite_architecture](https://github.com/biomed-AI/MetalSite/blob/main/image/MetalSite_architecture.jpg)

# System requirement
MetalSite is developed under Linux environment with:  
python  3.8.5  
numpy  1.19.1  
pandas  1.1.3  
torch  1.7.1  
sentencepiece  0.1.96  
transformers  4.17.0  
tqdm  4.59.0  

# Pretrained language model
You need to prepare the pretrained language model ProtTrans to run MetalSite:
1. Download the pretrained ProtT5-XL-UniRef50 model ([guide](https://github.com/agemagician/ProtTrans)). # ~ 11.3 GB (download: 5.3 GB)
2. Set path variable `ProtTrans_path` in `./script/MetalSite_predict.py`.

# Run MetalSite for prediction
Simply run:
```
python ./script/MetalSite_predict.py --fasta ./example/demo.fa --outpath ./example/
```
And the prediction results will be saved in `./example/demo_predictions.csv`. We also provide the corresponding canonical prediction results in `./example/demo_predictions_ref.csv` for your reference.

Other parameters:
```
--feat_bs       Batch size for ProtTrans feature extraction, default=10
--pred_bs       Batch size for MetalSite prediction, default=16
--save_feat     Save intermediate ProtTrans features
--gpu           Use GPU for feature extraction and MetalSite prediction
```

# Dataset and model
We provide the datasets and the trained MetalSite models here for those interested in reproducing our paper.  
The metal ion datasets used in this study are stored in `./datasets/` in fasta format.
The trained MetalSite models from 5-fold cross-validation can be found under `./model/`.

# Contact
Qianmu Yuan (yuanqm3@mail2.sysu.edu.cn)  
Yuedong Yang (yangyd25@mail.sysu.edu.cn)
