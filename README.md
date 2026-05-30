# CAD-ST

CAD-ST is a deep learning framework for histology-based spatial gene expression prediction. The model integrates histological image embeddings, spatial coordinate information, context-aware Transformer modeling, distance-aware soft contrastive learning, mask-based context reconstruction, and dual-stream graph refinement to predict spatial gene expression profiles from tissue morphology and spatial structure.

The current implementation supports experiments on the HER2ST breast cancer spatial transcriptomics dataset and the human cutaneous squamous cell carcinoma cSCC spatial transcriptomics dataset.

---

## Overview

Spatial transcriptomics provides gene expression measurements with spatial coordinates, but generating high-quality spatial transcriptomic profiles remains costly and experimentally demanding. CAD-ST aims to predict spatial gene expression from histological image-derived features and spatial information.
<img width="12930" height="7770" alt="fig 1" src="https://github.com/user-attachments/assets/29c8750c-9429-4106-8b65-97c50a40f6e8" />

The framework contains the following main components:

- **Histology feature encoder** for processing pre-extracted pathology image embeddings.
- **Gene feature encoder** for mapping spatial gene expression profiles into a latent space.
- **Spatial coordinate embedding** for incorporating spot-level spatial positions.
- **Context encoder** based on Transformer blocks for modeling tissue-level contextual dependencies.
- **Distance-aware soft contrastive learning** to align image-derived representations and gene representations under spatially smoothed supervision.
- **Mask-based context reconstruction** to enhance robustness by reconstructing masked spatial representations.
- **Dual-stream graph neural network** for combining physical spatial neighborhoods and semantic similarity neighborhoods.
- **Gene prediction head** for reconstructing spatial gene expression values.

---

## Project Structure

```text
CAD-ST/
├── herst.py              # Dataset loaders for HER2ST and cSCC
├── model.py              # CAD-ST model, Transformer blocks, DS-GNN, and loss functions
├── train.py              # Training script
├── predict.py            # Testing / prediction script
├── utils.py              # Argument parser, metric calculation, random seed utilities
├── requirements.txt      # Python dependencies
├── README.md             # Project description and usage instructions
└── data/                 # Dataset directory, not included in this repository

---

## Environment

The required environment can be installed from requirements.txt.

pip install -r requirements.txt

---

## Datasets
Human HER2-positive breast tumor ST data https://github.com/almaan/her2st/.
Human cutaneous squamous cell carcinoma 10x Visium data (GSE144240).

















