# GRAM-DTI: Adaptive Multimodal Representation Learning for Drug-Target Interaction Prediction

[![Venue: ICLR 2026](https://img.shields.io/badge/Venue-ICLR%202026-blue.svg)](https://iclr.cc/)
[![arXiv](https://img.shields.io/badge/arXiv-2509.21971-b31b1b.svg)](https://arxiv.org/abs/2509.21971)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation of the ICLR 2026 paper: **"GRAM-DTI: Adaptive Multimodal Representation Learning for Drug-Target Interaction Prediction"** ([arXiv:2509.21971](https://arxiv.org/abs/2509.21971)).

## 📖 Overview

Drug target interaction (DTI) prediction is a cornerstone of computational drug discovery. While deep learning has advanced DTI modeling, existing approaches primarily rely on pairwise SMILES-protein interactions, failing to exploit the rich multimodal information available for small molecules. 

**GRAM-DTI** is a novel pre-training framework that integrates four distinct modalities into unified representations:
1. **SMILES Sequences** (via MolFormer)
2. **Text Descriptions / Molecular Functions** (via MolT5)
3. **Hierarchical Taxonomic Annotations (HTA)** (via MolT5)
4. **Protein Sequences** (via ESM-2)

### Key Contributions
* **Gramian Volume-Based Multimodal Alignment:** Extends contrastive learning to four modalities, capturing higher-order semantic alignments beyond conventional pairwise approaches.
* **Gradient-Informed Adaptive Modality Dropout:** Dynamically regulates each modality's contribution during pre-training based on its gradient informativeness, preventing dominant but less informative modalities from overwhelming complementary signals.
* **Auxiliary Weak Supervision:** Incorporates IC50 activity measurements (when available) to ground learned representations in biologically meaningful interaction strengths.

## 📂 Repository Structure

```text
GRAM-DTI/
├── pretraining/
│   ├── __init__.py
│   ├── losses.py       # Implementation of Volume Loss, and Adaptive Dropout
│   ├── models.py       
│   ├── trainer.py      # Training loop, optimization, and logging logic
│   └── run.py          # Main entry point for distributed pre-training
├── README.md
└── requirements.txt    # Dependencies
```

## 💾 Data

The pre-training and downstream data are hosted on Hugging Face: [FengJiang02/GRAM-DTI](https://huggingface.co/datasets/FengJiang02/GRAM-DTI). The dataset card there describes every file and field.

| Path | Size | Description |
|---|---|---|
| `embeddings/embeddings.h5` | 256 MB | Pre-computed embeddings for the four modalities (50,968 molecule–protein pairs) |
| `embeddings/metadata.json` | 168 MB | Raw text and IC50 labels for the same pairs, in the same row order |
| `cleaned_pretrain_after_rdk.json` | 146 MB | Source records the embeddings were computed from |
| `downstream/` | 65 MB | Yamanishi_08, Hetionet, and the activation and inhibition MoA datasets, with `make_splits.py` |

### Pre-training data

Download it into `pretrain_data/`, the default `--embedding_dir` of `pretraining/run.py`:

```bash
hf download FengJiang02/GRAM-DTI --repo-type dataset \
  --include "embeddings/*" "cleaned_pretrain_after_rdk.json" --local-dir pretrain_data
```

### Downstream data

Download the four datasets into `data/`:

```bash
hf download FengJiang02/GRAM-DTI --repo-type dataset --include "downstream/*" --local-dir .
mv downstream data
```

Then generate the cross-validation folds for the warm-start, drug cold-start and protein cold-start settings:

```bash
python data/make_splits.py data/yamanishi_08 data/Hetionet
python data/make_splits.py data/activation data/inhibition --n_splits 5
```

The folds are written to `data/<dataset>/data_folds/<setting>/` and take about 15 GB in total.

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{jiang2025gramdti,
  title={GRAM-DTI: adaptive multimodal representation learning for drug target interaction prediction},
  author={Jiang, Feng and Mollaysa, Amina and Ma, Hehuan and Mansi, Tommaso and Huang, Junzhou and Prakash, Mangal and Liao, Rui},
  journal={arXiv preprint arXiv:2509.21971},
  year={2025}
}
```
