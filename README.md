# Neural Graph Generation with Conditioning
## ALTEGRAD 2024 Challenge Report
- **Authors**: Pierre Aguié, Valentin Dorseuil, Théo Molfessis
- **Emails**: {pierre.aguie, valentin.dorseuil, theo.molfessis}@polytechnique.edu
- **Date**: January 15, 2025

## Introduction
This repository contains the code for our submission to the ALTEGRAD 2024 Kaggle Challenge. The goal of this challenge is to perform conditional graph generation: given a set of properties ( c ) (e.g., number of nodes, number of edges, average degree, etc.), we aim to generate a graph that respects those properties as much as possible.

We first introduce the baseline provided by the organizers of the competition and highlight its limitations in the challenge’s setting. We then present our proposed baseline, based on a Conditional Variational Autoencoder (CVAE) architecture, and present the results of experiments showcasing the differences in performance of both methods, justifying our choices.

## Repository Structure
```
.
├── code
│   ├── train_NGG.py                # Training script for NGG
│   ├── sample_NGG.py               # Sampling script for NGG
│   ├── train_CVGAE.py              # Training script for CVGAE
│   ├── sample_CVGAE.py             # Sampling script for CVGAE
│   ├── train_contrastive_CVGAE.py  # Training script for Contrastive CVGAE
│   ├── sample_contrastive_CVGAE.py # Sampling script for Contrastive CVGAE
│   ├── visualize_latent.py         # Script to visualize the latent space
│   ├── utils
│   │   ├── data_processing.py      # Data preprocessing utilities
│   │   ├── extract_feats.py        # Feature extraction utilities
│   │   ├── eval.py                 # Evaluation metrics
│   │   ├── noise_schedules.py      # Noise schedules for diffusion models
│   │   └── visuals.py              # Visualization utilities
│   ├── model
│   │   ├── autoencoder.py          # Variational Autoencoder model
│   │   ├── cvae.py                 # Conditional Variational Autoencoder model
│   │   ├── contrastive_cvae.py     # Contrastive Conditional Variational Autoencoder model
│   │   └── denoise_model.py        # Denoising model for diffusion
├── data                            # Directory for datasets
├── models                          # Directory for saved models
├── outputs                         # Directory for output files
├── README.md                       # This README file
```

## Setup Instructions
Required python 3.8+
1. Install the required packages
```
pip install -r requirements.txt 
```

2. Train your model, CVGAE, NGG or ContrastiveCVGAE.
```python
python code/train_NGG.py --max-epochs 200 --batch-size 256
```

3. Sample graphs and look at the visuals
```python
python code/sample_NGG.py --batch-size 256
```


## Acknowledgments
We would like to thank the organizers of the ALTEGRAD 2024 Kaggle Challenge for providing the baseline and the dataset for this competition.

