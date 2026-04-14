# Project Roadmap

This roadmap breaks the challenge brief into executable phases, from setup to final submission.

## Current focus

Start with Phase 1 and Phase 2:

- Confirm the target track and metric definitions.
- Finalize the dataset layout.
- Prepare raw data and run preprocessing into `data/processed/`.

## Deliverable rule

Every phase should end with a visible artifact:

- code
- log file
- figure or table
- short write-up note
- exported checkpoint or zip, when applicable

## Phase 1: Scope and setup

- Confirm the final challenge track to target.
- Fix the evaluation metric definitions, especially FS.
- Finalize dataset layout and folder naming.
- Create the repository scaffold and install dependencies.
- Output: locked scope, folder structure, and a verified environment.

## Phase 2: Data preparation

- Collect raw MRI data and annotations.
- Organize raw data into detection, classification, segmentation, and GAN folders.
- Run preprocessing to resize, normalize, and copy images into `data/processed/`.
- Verify class balance, missing masks, and file naming consistency.
- Output: raw dataset zip, preprocessed dataset zip, and dataset summary note.

## Phase 3: Baseline models

- Train the detection CNN baseline.
- Train the U-Net segmentation baseline.
- Train the tumour type classifier baseline.
- Save baseline metrics for comparison.
- Output: baseline checkpoints, logs, confusion matrix, Dice, IoU, and F1.

## Phase 4: GAN development

- Build the generator and discriminator.
- Train the GAN on unlabeled MRI data.
- Save sample grids, checkpoints, and loss curves.
- Compute FID per epoch and store the history.
- Output: GAN checkpoints, sample images, history file, loss curve, and FID curve.

## Phase 5: GAN augmentation experiment

- Generate synthetic MRI samples.
- Augment the classifier training set with synthetic data.
- Re-train the classifier and compare against the baseline.
- Report the improvement in accuracy, F1, and any class-wise gains.
- Output: baseline-vs-augmented comparison table.

## Phase 6: Evaluation and reporting

- Calculate detection, segmentation, and classification metrics.
- Report FID and FS/F1 across epochs.
- Generate confusion matrices, Dice, IoU, and loss plots.
- Prepare the dataset write-up and innovation write-up.
- Output: final metrics summary and the two one-page write-ups.

## Phase 7: Deployment and packaging

- Wire the Streamlit app to saved artifacts.
- Export trained weights in Keras format.
- Package the raw and preprocessed dataset zips.
- Prepare the architecture diagram and final submission folder.
- Output: runnable Streamlit demo and submission-ready archive structure.

## Phase 8: Final submission check

- Confirm all deliverables are present.
- Confirm the folder name follows the roll number format.
- Confirm the write-ups are complete and consistent with the code.
- Verify the app launches from the repository root.
- Output: final checklist pass before submission.
