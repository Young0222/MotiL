### DIMO ###

This repository is the official implementation of **DIMO** for macromolecules, which is model proposed in a paper: [**Diffusion-primed molecular contrastive learning for micro- and macromolecules**]. 

We propose DIffusion-primed Molecular cOntrastive learning (DIMO), which integrates generative diffusion models with molecular contrastive learning.
DIMO includes three phases totally: 1) diffusion priming, 2) multi-scaled contrastive learning with network pruning, and 3) task-specific finetuning

# Decompress data
tar -xzvf go.tar.gz
tar -xzvf ec.tar.gz

# Install environment
pip install -r requirements.txt

# Running model
bash run.sh
