
### Diffusion-primed molecular contrastive learning for micro- and macromolecules ###

This repository is the official implementation of **DIMO** for micromolecules, which is model proposed in a paper: [**Diffusion-primed molecular contrastive learning for micro- and macromolecules**]. 

We propose DIffusion-primed Molecular cOntrastive learning (**DIMO**), which integrates generative diffusion models with molecular contrastive learning.

## 🤖 Model

DIMO includes three phases totally: 1) diffusion priming, 2) multi-scaled contrastive learning with network pruning, and 3) task-specific finetuning.

**Diffusion priming**: The diffusion molecular model **DIMO** enables the GNN encoder to accurately capture the molecular structure before contrastive learning.

**Multi-scaled contrastive learning with network pruning**: We make the entire contrastive learning phase augmentation-free by replacing molecule graph augmentation with network pruning to generate two distinct representations. Additionally, we design the multi-scaled contrastive mechanism to ensure consistency between representations from the same molecule or functional group.

**Task-specific fine-tuning**: The pretrained GNN encoders can be used to predict the representations of unseen molecules, which are then inputted into an MLP classifier to learn the relationship between graph representations and the ground truth molecular property labels.

# Install environment
pip install -r requirements.txt

# Running DIMO for pretraining (diffusion priming and contrastive learning)
bash pretrain_dimo.sh

# Running DIMO for finetuning
bash finetune_dimo.sh