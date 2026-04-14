# Phase 7: Deployment and Packaging

## Goal

Produce a runnable app and a submission-ready archive.

## Deployment tasks

- Keep `streamlit_app.py` artifact-aware.
- Confirm app starts from repository root.
- Keep architecture and metrics docs aligned with code outputs.

## Packaging tasks

- Build final folder using roll-number naming.
- Bundle code, docs, artifacts, and dataset zips.
- Export final zip for submission upload.

## Commands

```bash
streamlit run streamlit_app.py
python build_submission_package.py --roll 2300030497 --name "PALADAGULA VENKATA RAJEEV"
```
