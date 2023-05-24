# DeepGate2: Functionality-Aware Circuit Representation Learning

Code repository for the paper:  
**DeepGate2: Functionality-Aware Circuit Representation Learning**, 

Zhengyuan Shi, Hongyang Pan, Sadaf Khan, Min Li, Yi Liu, Junhua Huang, Hui-Ling Zhen, Mingxuan Yuan, Zhufei Chu and Qiang Xu

## Abstract 
Circuit representation learning aims to obtain neural representations of circuit elements and has emerged as a promising research direction that can be applied to various EDA and logic reasoning tasks. Existing solutions, such as DeepGate, have the potential to embed both circuit structural information and functional behavior. However, their capabilities are limited due to weak supervision or flawed model design, resulting in unsatisfactory performance in downstream tasks. In this paper, we introduce **DeepGate2**, a novel functionality-aware learning framework that significantly improves upon the original DeepGate solution in terms of both learning effectiveness and efficiency. Our approach involves using pairwise truth table differences between sampled logic gates as training supervision, along with a well-designed and scalable loss function that explicitly considers circuit functionality. Additionally, we consider inherent circuit characteristics and design an efficient one-round graph neural network (GNN), resulting in an order of magnitude faster learning speed than the original DeepGate solution. Experimental results demonstrate significant improvements in two practical downstream tasks: logic synthesis and Boolean satisfiability solving.

## Installation
```sh
conda create -n deepgate2 python=3.8.9
conda activate deepgate2
pip install -r requirements.txt
```

## Prepare Dataset 
Please refer the script `./src/prepare_dataset.py` for dataset preparation. You should set the folder with circuit netlist in .bench format. 
```sh
cd src
python ./src/prepare_dataset.py # Use the default settings
```

## Model Training 
The model training is separated into two stages. 
Model learns to predict the logic probability in Stage.1, which is the same task with previous version of DeepGate
Model learns to predict the pairwise truth table difference of two gates in Stage.2. 
```sh
bash ./run/stage1_train.sh
bash ./run/stage2_trun.sh
```
## Model Inference
You can test the model with the testing dataset. Please remember generate the testing dataset before that. 
```sh
bash ./run/test.sh
```

This model also support to parse raw data (in .aig or .bench format) and generate gate embeddings into file. Therefore, you can apply these embeddings to downstream tasks
```sh
cd src
python get_emb_bench.py
```


