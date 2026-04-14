from __future__ import annotations

import json
from pathlib import Path


def count_files(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.rglob("*") if p.is_file())


def summarize_folder(folder: Path):
    summary = {}
    if not folder.exists():
        return summary

    for path in sorted(folder.iterdir()):
        if path.is_dir():
            summary[path.name] = count_files(path)
    return summary


def main():
    processed_root = Path("data/processed")
    report = {
        "detection": summarize_folder(processed_root / "detection"),
        "classification": summarize_folder(processed_root / "classification"),
        "segmentation": {
            "images": count_files(processed_root / "segmentation" / "images"),
            "masks": count_files(processed_root / "segmentation" / "masks"),
        },
        "gan_images": count_files(processed_root / "gan_images"),
    }

    output_path = Path("artifacts/dataset_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
