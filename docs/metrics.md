# Evaluation Notes

## Challenge scoring focus

The submission should be documented around the scoring categories in the challenge brief:

- Innovation
- Deliverables
- Innovation write-up
- FID and FS performance
- Dataset write-up
- GAN code
- Classifier code

## FID

Frechet Inception Distance is used to compare the distribution of real MRI images and generated MRI images in feature space. Lower values indicate better sample quality.

## FS

The challenge draft refers to FS without defining it. This project treats FS as F1-score unless the faculty provides a different definition from the faculty.

## What to report

For a complete submission, keep these artifacts together:

- `artifacts/detection/` detection checkpoints and logs
- `artifacts/segmentation/` segmentation checkpoints and logs
- `artifacts/classifier/` classifier checkpoints and logs
- `artifacts/gan/` GAN checkpoints, sample images, history, FID curve, and loss curve

## What the faculty will likely inspect

- Whether the GAN actually improves downstream classification
- Whether the synthetic images look plausible and diverse
- Whether the write-up explains the data sources and preprocessing clearly
- Whether metrics are reported per epoch and compared with a baseline

## Recommended reporting

- Detection: accuracy, AUC, precision, recall, F1
- Segmentation: Dice, IoU, pixel accuracy
- GAN: FID, FS/F1, loss curves, visual samples
