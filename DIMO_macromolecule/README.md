# DIMO #

This repository is the official implementation of **DIMO** for macromolecules, which is proposed in a paper: [**Diffusion-primed molecular contrastive learning for micro- and macromolecules**]. 

We propose **DI**ffusion-primed **M**olecular c**O**ntrastive learning (**DIMO**), which integrates generative diffusion models with molecular contrastive learning.
DIMO includes three phases totally: 1) diffusion priming, 2) multi-scaled contrastive learning with network pruning, and 3) task-specific finetuning

# Download data and pretrained model
Before running the model, please download the data and pretrained model from this URL: https://pan.baidu.com/s/1JpsVM3WhISKdmWWlU0fhew?pwd=ybbw

# Decompress data
mkdir data

tar -xzvf go.tar.gz

tar -xzvf ec.tar.gz

mv go ec data

# Install environment
pip install -r requirements.txt

# Running model for pretraining (diffusion priming and contrastive learning)
bash pretrain_dimo.sh

# Running model for finetuning
bash finetune_dimo.sh
