# Phase 2: Data Preparation

## Goal

Turn raw MRI assets into the processed layout used by the training scripts.

## Inputs

- Raw MRI images
- Labels for detection and classification
- Segmentation images and masks
- GAN-only unlabeled MRI images

## Outputs

- `data/processed/detection/normal`
- `data/processed/detection/tumour`
- `data/processed/classification/<class_name>`
- `data/processed/segmentation/images`
- `data/processed/segmentation/masks`
- `data/processed/gan_images`
- Dataset summary note

## Checks

- No missing segmentation masks
- Consistent image naming across paired files
- Class counts are visible before training
- Resized images match the model input sizes
