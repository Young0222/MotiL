
# Molecular Motif Learning #

This repository is the official implementation of **MotiL** for micromolecules, which is proposed in a paper: [**Molecular Motif Learning**]. 

We propose **Mo**lecular mo**ti**f **L**earning (**MotiL**), which is a novel pretraining representation learning method for molecules.

## Brief introduction

MotiL includes three phases totally: 1) diffusion priming, 2) bi-scaled training, and 3) task-specific fine-tuning.

**Diffusion priming**: The diffusion molecular model **MotiL** enables the GNN encoder to accurately capture the molecular structure before contrastive learning.

**Bi-scaled training**: We make the entire bi-scaled training phase augmentation-free by using network pruning to generate two distinct representations. Additionally, we ensure the consistency between representations from the same graph (i.e., small molecule or protein) or motif (i.e., functional group or amino acid).

**Task-specific fine-tuning**: The pretrained GNN encoders can be used to predict the representations of unseen molecules, which are then inputted into an MLP classifier to learn the relationship between graph representations and the ground truth molecular property labels.

# Install environment
pip install -r requirements.txt

# Running MotiL
**For pretraining (diffusion priming and bi-scaled training)**:

bash pretrain_motil.sh

**For finetuning**:

bash finetune_motil.sh
