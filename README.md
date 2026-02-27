# GRAM-DTI: Adaptive Multimodal Representation Learning for Drug-Target Interaction Prediction

[![Venue: ICLR 2026](https://img.shields.io/badge/Venue-ICLR%202026-blue.svg)](https://iclr.cc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation of the ICLR 2026 paper: **"GRAM-DTI: Adaptive Multimodal Representation Learning for Drug-Target Interaction Prediction"**.

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