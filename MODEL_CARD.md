# Model card: 2.5D U-Net for GI tract segmentation

> **Research demonstration only. This model is not a medical device, has not been clinically validated,
> and must not be used for diagnosis, treatment planning or any clinical decision.**

## What it does
For each abdominal MRI slice, it predicts masks for three organs: the **large bowel**, the **small bowel**
and the **stomach**.

The input is three neighbouring slices (z − 2, z, z + 2) stacked as channels ("2.5D"). That gives the 2D
network some depth context at almost no extra cost. The architecture is a U-Net with a ResNet-34 encoder
pretrained on ImageNet.

## Context
In MR-guided radiotherapy, the bowel and stomach must be outlined on every session so radiation can avoid
them. Done by hand, this takes a long time. Automatic outlining is the research problem behind the
UW-Madison GI Tract Image Segmentation competition (Kaggle).

## Training data
UW-Madison GI Tract Image Segmentation (Kaggle), used under the competition's non-commercial and academic
terms:
- 274 MRI sessions from 85 patients, 38,496 slices with run-length-encoded organ masks.
- 16,590 slices contain at least one organ.

The data is not redistributed, and no scans or masks are included in this repository.

## Evaluation
- **Validation:** one of 5 folds, grouped by patient, so no patient appears in both training and
  validation.
- **Metric:** the competition score, 0.4 × Dice + 0.6 × (1 − normalised 3D Hausdorff distance), computed
  at each session's original resolution.
- **Baseline:** "predict nothing". It scores highly on Dice, because most slices are empty, so the
  Hausdorff term is where a model has to prove itself.

Results are in [RESULTS.md](RESULTS.md), generated from MLflow.

## Known limitations
- **One fold.** It is trained and validated on a single patient-grouped fold. Scores on other folds, or on
  other hospitals, scanners and protocols, will differ.
- **Low resolution.** Images are downsampled to 256 px, so very thin structures (small-bowel loops) lose
  detail.
- **No 3D consistency.** Each slice is predicted from three slices, so masks can flicker between
  neighbouring slices.
- **Fixed threshold.** The 0.5 threshold is not tuned per organ.
- **Narrow population.** A single centre and a single cohort: performance on other populations is unknown.

## Monitoring
`python -m gisseg.monitoring` compares incoming scans with the training reference on three checks:
- **intensity drift:** an alert when the PSI of mean slice intensity exceeds 0.2;
- **unseen resolutions:** an alert when more than 20% of volumes have an image size never seen in
  training;
- **organ prevalence:** an alert when the share of slices containing an organ changes by more than 0.15,
  measured on audited labels.

Any alert recommends retraining and review.
