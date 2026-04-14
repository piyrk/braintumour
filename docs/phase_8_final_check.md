# Phase 8: Final Submission Check

## Goal

Verify that all required components exist before creating the final submission archive.

## Automated check

Run:

```bash
python validate_submission.py
```

If training has not been run yet, use:

```bash
python validate_submission.py --no-training
```

This checks for:

- core scripts
- Streamlit app
- architecture and reporting documents
- package builder
- minimum artifact presence (checkpoint, log, and image)

In `--no-training` mode, the artifact presence checks are skipped.

## Final package command

```bash
python build_submission_package.py --roll 2300030497 --name "PALADAGULA VENKATA RAJEEV"
```

## Manual confirmation

- Folder naming follows roll-number format
- Write-ups are consistent with the code and outputs
- Streamlit app launches from repository root
