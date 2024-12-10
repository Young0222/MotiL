# MotiL #

This repository is the official implementation of **MotiL** for macromolecules, which is proposed in a paper: [**Diffusion-primed molecular contrastive learning for micro- and macromolecules**]. 

We propose **Mo**lecular mo**ti**f **L**earning (**MotiL**), which is a novel pretraining representation learning method for molecules.
MotiL includes three phases totally: 1) diffusion priming, 2) bi-scaled training, and 3) task-specific fine-tuning

# Download data and pretrained model
Before running the model, please download the data and pretrained model from this URL: https://pan.baidu.com/s/1JpsVM3WhISKdmWWlU0fhew?pwd=ybbw

# Decompress data
mkdir data

tar -xzvf go.tar.gz

tar -xzvf ec.tar.gz

mv go ec data

# Install environment
pip install -r requirements.txt

# Running DIMO
**For pretraining (diffusion priming and bi-scaled training)**:

bash pretrain_motil.sh

**For finetuning**:

bash finetune_motil.sh
