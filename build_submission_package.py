from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import zipfile


def parse_args():
    parser = argparse.ArgumentParser(description="Build submission package folder and zip archive.")
    parser.add_argument("--roll", required=True, help="Roll number for final folder naming.")
    parser.add_argument("--name", required=True, help="Student name for final folder naming.")
    parser.add_argument("--output-dir", default="submission", help="Base output directory.")
    return parser.parse_args()


def safe_copy(src: Path, dst: Path):
    if not src.exists():
        return
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def zip_folder(folder: Path, zip_path: Path):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in folder.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(folder))


def main():
    args = parse_args()
    folder_name = f"{args.roll}-{args.name}".upper()
    base_output = Path(args.output_dir)
    submission_root = base_output / folder_name

    if submission_root.exists():
        shutil.rmtree(submission_root)
    submission_root.mkdir(parents=True, exist_ok=True)

    safe_copy(Path("src"), submission_root / "src")
    safe_copy(Path("streamlit_app.py"), submission_root / "streamlit_app.py")
    safe_copy(Path("preprocess_data.py"), submission_root / "preprocess_data.py")
    safe_copy(Path("dataset_report.py"), submission_root / "dataset_report.py")
    safe_copy(Path("compare_runs.py"), submission_root / "compare_runs.py")
    safe_copy(Path("requirements.txt"), submission_root / "requirements.txt")
    safe_copy(Path("docs"), submission_root / "docs")
    safe_copy(Path("artifacts"), submission_root / "artifacts")

    raw_zip = submission_root / "raw_dataset.zip"
    processed_zip = submission_root / "processed_dataset.zip"

    if Path("data/raw").exists():
        zip_folder(Path("data/raw"), raw_zip)
    if Path("data/processed").exists():
        zip_folder(Path("data/processed"), processed_zip)

    final_zip = base_output / f"{folder_name}.zip"
    zip_folder(submission_root, final_zip)

    print(f"Created submission folder: {submission_root}")
    print(f"Created submission zip: {final_zip}")


if __name__ == "__main__":
    main()
