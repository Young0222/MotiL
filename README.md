![](https://img.shields.io/badge/version-1.0.0-blue)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/Young0222/MotiL/blob/main/LICENSE.txt)

# MotiL

> **Mo**lecular mo**ti**f **L**earning for molecular property prediction
>
> A motif-centered pretraining framework for molecular property prediction

This repository is the official implementation of **MotiL** for micro- and macromolecules, which is proposed in our [paper](https://www.nature.com/articles/s41467-025-66685-w).

MotiL is a molecular pretraining framework centered on **motif learning**.  
This repository contains two parts:

- `MotiL_micromolecule`: code for small-molecule pretraining and downstream prediction
- `MotiL_macromolecule`: code for macromolecule pretraining and downstream prediction

## ✨ Overview

MotiL learns molecular representations through three main stages:

1. **Diffusion priming**
2. **Bi-scaled training**
3. **Task-specific fine-tuning**

The goal is to learn better molecular representations before downstream property prediction, so that the model can perform well on both micromolecule and macromolecule tasks.

## 🧠 Model Introduction

![Overview of MotiL](motil.png)

As shown above, MotiL is built around a GNN encoder and learns molecular representations from both whole-molecule structure and motif-level patterns. In the first stage, **diffusion priming** perturbs and reconstructs molecular bonds to warm up the encoder and help it capture global structural information. In the second stage, **bi-scaled training** aligns representations at both the graph level and the motif level, so molecules with similar scaffolds and motifs with similar chemical meanings stay close in the representation space. In the final stage, the pretrained encoder is **fine-tuned** with task labels for downstream molecular property prediction on both micromolecules and macromolecules.

## 🗂 Repository Structure

```text
MotiL-main/
├── README.md
├── LICENSE.txt
├── MotiL_micromolecule/
│   ├── README.md
│   ├── requirements.txt
│   ├── pretrain.py
│   ├── train.py
│   ├── pretrain_motil.sh
│   ├── finetune_motil.sh
│   ├── data/
│   ├── dumped/
│   └── chemprop/
└── MotiL_macromolecule/
    ├── README.md
    ├── requirements.txt
    ├── datasets.py
    ├── models.py
    ├── modules.py
    ├── go.py
    ├── ec.py
    ├── pretrain_motil.sh
    └── finetune_motil.sh
```

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Young0222/MotiL.git
cd MotiL
```

### 2. Install dependencies

For micromolecule experiments:

```bash
cd MotiL_micromolecule
pip install -r requirements.txt
```

For macromolecule experiments:

```bash
cd ../MotiL_macromolecule
pip install -r requirements.txt
```

> Note  
> Some package entries in the provided `requirements.txt` files come from the original environment and may be platform-specific.  
> If you build a fresh environment, you may need to adjust a few package versions based on your local system.

## 🔬 Micromolecule

This part is the official implementation of **MotiL** for micromolecules.

We propose **Molecular motif Learning (MotiL)**, a novel pretraining representation learning method for molecules.

### Brief introduction

MotiL includes three phases:

1. **Diffusion priming**
2. **Bi-scaled training**
3. **Task-specific fine-tuning**

**Diffusion priming**  
The diffusion molecular model in MotiL enables the GNN encoder to accurately capture molecular structure before contrastive learning.

**Bi-scaled training**  
This phase is augmentation-free. MotiL uses network pruning to generate two distinct representations, and also enforces consistency between representations from the same graph, such as a small molecule or protein, or the same motif, such as a functional group or amino acid.

**Task-specific fine-tuning**  
The pretrained GNN encoders are used to predict representations for unseen molecules. These representations are then fed into an MLP classifier to learn the relationship between graph representations and ground-truth molecular property labels.

### Included resources

- Benchmark datasets are already provided in `MotiL_micromolecule/data/`
- A pretrained checkpoint is already included in `MotiL_micromolecule/dumped/pre-train/1-model/`

### Install environment

```bash
pip install -r requirements.txt
```

### Running MotiL

For pretraining, including diffusion priming and bi-scaled training:

```bash
cd MotiL_micromolecule
bash pretrain_motil.sh
```

The script runs:

```bash
python pretrain.py --exp_name 'pre-train' --exp_id 1 --step pretrain --gpu 5
```

For finetuning:

```bash
cd MotiL_micromolecule
bash finetune_motil.sh
```

The provided fine-tuning script includes example commands for:

| Task | Type |
|------|------|
| BBBP | Classification |
| ClinTox | Classification |
| BACE | Classification |
| ESOL | Regression |

If needed, please change the `--gpu` setting in the script to match your machine.

## 🧫 Macromolecule

This part is the official implementation of **MotiL** for macromolecules.

We propose **Molecular motif Learning (MotiL)**, a novel pretraining representation learning method for molecules.

MotiL includes three phases:

1. **Diffusion priming**
2. **Bi-scaled training**
3. **Task-specific fine-tuning**

### Step 1. Download data and pretrained files

Before running macromolecule experiments, please download the required data and pretrained files from:

[https://huggingface.co/datasets/Young0222/MotiL/tree/main](https://huggingface.co/datasets/Young0222/MotiL/tree/main)

### Step 2. Prepare data

```bash
cd MotiL_macromolecule
mkdir -p data
tar -xzvf go.tar.gz
tar -xzvf ec.tar.gz
mv go ec data
```

### Install environment

```bash
pip install -r requirements.txt
```

### Running MotiL

For pretraining, including diffusion priming and bi-scaled training:

```bash
cd MotiL_macromolecule
bash pretrain_motil.sh
```

This script contains example commands for:

- GO dataset
- EC dataset

For finetuning:

```bash
cd MotiL_macromolecule
bash finetune_motil.sh
```

This script contains example commands for:

- GO-CC
- EC

If needed, please change the `--gpu` setting in the script before running.

## 📊 Reproducibility

The original README content is kept below.

**After running the provided scripts, you will get the following logs where the fine-tuned results align with those reported in our paper.**

Examples:

BBBP [MotiL_micromolecule/nohup.out.bbbp](MotiL_micromolecule/nohup.out.bbbp)

ClinTox [MotiL_micromolecule/nohup.out.clintox](MotiL_micromolecule/nohup.out.clintox)

BACE [MotiL_micromolecule/nohup.out.bace](MotiL_micromolecule/nohup.out.bace)

ESOL [MotiL_micromolecule/nohup.out.esol](MotiL_micromolecule/nohup.out.esol)

GO-CC [MotiL_macromolecule/nohup.out.cc](MotiL_macromolecule/nohup.out.cc)

EC [MotiL_macromolecule/nohup.out.ec](MotiL_macromolecule/nohup.out.ec)

## 📖 Citation

If you find this repository useful in your research, please cite our [paper](https://www.nature.com/articles/s41467-025-66685-w).

```bibtex
@article{liu2025motil,
  title={Molecular Motif Learning as a pretraining objective for molecular property prediction},
  author={Liu, Ziyang and Wang, Chaokun and Zheng, Shuwen and Wu, Cheng and Feng, Hao and Xu, Li and Zheng, Yue and Rong, Liang and Li, Peng},
  journal={Nature Communications},
  year={2025},
  doi={10.1038/s41467-025-66685-w}
}
```

## 📄 License

This project is released under the MIT License. See [LICENSE.txt](LICENSE.txt) for details.
