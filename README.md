# CAD-ST

CAD-ST is a deep learning framework for spatial gene expression prediction from histological image representations and spatial transcriptomics data. The model integrates image-derived spot features, spatial coordinates, local contextual attention, distance-aware soft contrastive learning, masked reconstruction, and a dual-stream graph neural network to improve spatial gene expression reconstruction across tissue sections.

CAD-ST is designed for spatial transcriptomics datasets where each tissue spot is associated with gene expression profiles, spatial coordinates, and histology-derived image embeddings. The current implementation supports the HER2ST breast cancer dataset and the human cutaneous squamous cell carcinoma dataset from GSE144240.

<img width="12930" height="7770" alt="fig 1" src="https://github.com/user-attachments/assets/d2e5a8a2-6e53-46d1-a4df-ac9e268ef4e0" />


## Overview

The main components of CAD-ST include:

- **Image feature encoder** for projecting histology-derived spot embeddings into the latent space.
- **Gene encoder** for learning gene-expression latent representations.
- **Spatial coordinate embedding** for incorporating spot-level spatial positions.
- **Local context Transformer** for modeling neighborhood-aware spatial dependencies.
- **Distance-aware soft contrastive learning** to align image and gene representations with spatially smoothed soft targets.
- **Spatial block masking and reconstruction** to improve local representation robustness.
- **Dual-stream graph neural network** to jointly model physical spatial neighborhoods and semantic feature relationships.
- **Gene expression decoder** for reconstructing spatial gene expression profiles.

## Environment

The code was developed with Python and PyTorch. The required dependencies can be installed using:

```bash
pip install -r requirements.txt
```

## Datasets

- Human HER2-positive breast tumor ST data https://github.com/almaan/her2st/.
- Human cutaneous squamous cell carcinoma 10x Visium data (GSE144240).
- The GSE175540 Visium FFPE dataset used for external TLS-associated immune region analysis.

## Training

### Train CAD-ST on one fold

For HER2ST:

```
python train.py \
  --dataset her2st \
  --fold 0 \
  --no-all_folds \
  --device_id 0 \
  --output_dir output
```

For cSCC:

```
python train.py \
  --dataset cSCC \
  --fold 0 \
  --no-all_folds \
  --device_id 0 \
  --output_dir output
```

### Train CAD-ST on all folds

For HER2ST:

```
python train.py \
  --dataset her2st \
  --all_folds \
  --device_id 0 \
  --output_dir output
```

For cSCC:

```
python train.py \
  --dataset cSCC \
  --all_folds \
  --device_id 0 \
  --output_dir output
```

## Prediction

After training, run prediction or testing using:

```
python predict.py \
  --dataset cSCC \
  --fold 0 \
  --device_id 0 \
  --output_dir output
```

or for HER2ST:

```
python predict.py \
  --dataset her2st \
  --fold 0 \
  --device_id 0 \
  --output_dir output
```

