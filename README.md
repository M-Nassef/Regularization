# Regularization

This lab exercise compares regularization techniques on a small CNN that classifies
**Dogs vs. Cats** images (Keras / TensorFlow, Google Colab, T4 GPU).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/M-Nassef/Regularization/blob/main/Lab09_1_Regularization.ipynb)

## Setup

- **Data:** the Kaggle Dogs vs. Cats training set (25,000 images), split 80 / 20 into
  20,000 training and 5,000 validation images, resized to 150 × 150.
- **Baseline model:** 3 × (Conv2D + MaxPooling) with 32, 64 and 128 filters, then
  Dense(128) and a sigmoid output. Trained with Adam and binary cross-entropy for 10 epochs.
- **Evaluation:** validation accuracy, a confusion matrix, and examples of misclassified images.

## Experiments

Each experiment is a separate commit, so you can open any version from the commit history.
The table shows validation accuracy from a single 10-epoch run of each notebook version.

| Commit | Change | Best val. accuracy | Final val. accuracy |
|---|---|---|---|
| `f776b0c` | Baseline, no regularization | 0.834 | 0.808 |
| `60da34f` | Dropout (0.5, 0.5, 0.3) | 0.813 | 0.810 |
| `3f01485` | 2 × L2 (0.001) + 1 × Dropout (0.5) | **0.838** | 0.820 |
| `8158715` | 3 × L2 (0.001) + 1 × Dropout (0.5) | 0.836 | **0.831** |
| `0f98887` | 3 × L2 (0.001) + 2 × Dropout (0.5) | 0.750 | 0.745 |
| `5b56d2b` | All techniques together | 0.794 | 0.778 |
| `89d5eec` | Baseline + early stopping (patience 3, restore best weights) | 0.835 | 0.832 (stopped after 6 epochs) |

**Observations**
- The baseline overfits quickly. Training accuracy reaches about 95 % while validation
  accuracy stays at about 83 % and validation loss starts rising after epoch 3.
- Moderate L2 plus a single dropout layer gave the most stable validation accuracy.
- Too much regularization (L2 plus two dropout layers, or everything combined) underfits
  within 10 epochs.
- Early stopping gives about the same accuracy as the best regularized models at a lower
  training cost, because it stops when validation loss stops improving.

These are single runs without fixed random seeds, so differences of about 1 % are within run-to-run noise.

## Running it

Open the notebook in Colab with the badge above and choose a **T4 GPU** runtime.
The notebook downloads the dataset (about 850 MB) itself with `gdown`.
