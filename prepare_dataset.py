from __future__ import annotations

from pathlib import Path
import shutil


def ensure_structure(root: Path):
    for path in [
        root / "processed" / "detection" / "normal",
        root / "processed" / "detection" / "tumour",
        root / "processed" / "classification",
        root / "processed" / "segmentation" / "images",
        root / "processed" / "segmentation" / "masks",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def summarize(root: Path):
    print("Dataset layout:")
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            print(path)


def main():
    root = Path("data")
    ensure_structure(root)
    print("Created/verified expected dataset folders.")
    summarize(root)
    print("Place raw MRI data in data/raw and copy or preprocess it into data/processed.")


if __name__ == "__main__":
    main()
