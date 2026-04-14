# Submission Checklist

Use this before handing the project in.

## Code

- [ ] `train_detection.py`
- [ ] `train_segmentation.py`
- [ ] `train_classifier.py`
- [ ] `train_gan.py`
- [ ] `streamlit_app.py`
- [ ] `preprocess_data.py`

## Data

- [ ] Raw dataset zip prepared
- [ ] Preprocessed dataset zip prepared
- [ ] Folder naming follows roll-number format

## Model outputs

- [ ] Best checkpoints saved in Keras format
- [ ] Pretrained weights exported
- [ ] Training logs exported
- [ ] Loss curves exported
- [ ] Confusion matrices exported
- [ ] FID and FS/F1 curves exported

## Write-up

- [ ] One-page dataset note
- [ ] One-page innovation note
- [ ] Architecture diagram
- [ ] Deployment note for Streamlit

## Final validation

- [ ] `python validate_submission.py` passes (or `python validate_submission.py --no-training` before training)
- [ ] Final package zip created with `build_submission_package.py`

