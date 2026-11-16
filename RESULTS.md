# Results

Generated from MLflow on 2026-11-15 22:39 UTC by `python -m gisseg.report`.
Metric: the competition score, **0.4 x Dice + 0.6 x (1 - normalised 3D Hausdorff)** (higher is better),
computed at each volume's original resolution on validation fold 0 of 5:
55 MRI sessions from patients never seen in training (GroupKFold by patient).

| Model | Score | Dice | Hausdorff (normalised, lower is better) | MLflow run |
|---|---:|---:|---:|---|
| 2.5D U-Net, resnet34 encoder | **0.9340** | 0.9108 | 0.0506 | `4ba37f9a1ba2419c8ae77e7765744057` |
| Predict nothing (baseline) | 0.2802 | 0.7005 | 1.0000 | `3e9ebac1e932426e929312e93bc3eb8b` |

Predicting nothing already scores well on Dice, because most slices contain no organ and an empty
prediction on an empty slice counts as perfect. The Hausdorff term is what separates real models.

## By organ

| Organ | Score | Dice | Hausdorff |
|---|---:|---:|---:|
| large bowel | 0.9272 | 0.9041 | 0.0574 |
| small bowel | 0.9083 | 0.8700 | 0.0661 |
| stomach | 0.9664 | 0.9584 | 0.0283 |

## Training (from the run's parameters and history)

- Unet with a resnet34 encoder (imagenet weights), input 256 px, 2.5D context stride 2
- 15 epochs, batch 16, peak learning rate 0.001, mixed precision True, on NVIDIA GeForce RTX 3050 6GB Laptop GPU
- Best epoch 14 (validation Dice at 256 px 0.9124); total time 131 minutes
