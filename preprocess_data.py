from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def ensure_structure(processed_root: Path):
    for path in [
        processed_root / "detection" / "normal",
        processed_root / "detection" / "tumour",
        processed_root / "classification",
        processed_root / "segmentation" / "images",
        processed_root / "segmentation" / "masks",
        processed_root / "gan_images",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def resize_and_save_image(source_path: Path, destination_path: Path, size: tuple[int, int], grayscale: bool = True):
    image = Image.open(source_path)
    image = image.convert("L" if grayscale else "RGB")
    image = image.resize(size)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination_path)


def copy_image(source_path: Path, destination_path: Path):
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)


def iter_images(folder: Path):
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def preprocess_detection(raw_root: Path, processed_root: Path, size: tuple[int, int]):
    source_root = raw_root / "detection"
    destination_root = processed_root / "detection"
    counts = {}

    for class_name in ("normal", "tumour"):
        source_dir = source_root / class_name
        destination_dir = destination_root / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        counts[class_name] = 0
        if not source_dir.exists():
            continue
        for image_path in iter_images(source_dir):
            destination_path = destination_dir / f"{image_path.stem}.png"
            resize_and_save_image(image_path, destination_path, size, grayscale=True)
            counts[class_name] += 1

    return counts


def preprocess_classification(raw_root: Path, processed_root: Path, size: tuple[int, int]):
    source_root = raw_root / "classification"
    destination_root = processed_root / "classification"
    counts = {}

    if not source_root.exists():
        return counts

    for class_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        destination_dir = destination_root / class_dir.name
        destination_dir.mkdir(parents=True, exist_ok=True)
        counts[class_dir.name] = 0
        for image_path in iter_images(class_dir):
            destination_path = destination_dir / f"{image_path.stem}.png"
            resize_and_save_image(image_path, destination_path, size, grayscale=True)
            counts[class_dir.name] += 1

    return counts


def preprocess_segmentation(raw_root: Path, processed_root: Path, size: tuple[int, int]):
    image_source = raw_root / "segmentation" / "images"
    mask_source = raw_root / "segmentation" / "masks"
    image_destination = processed_root / "segmentation" / "images"
    mask_destination = processed_root / "segmentation" / "masks"
    image_destination.mkdir(parents=True, exist_ok=True)
    mask_destination.mkdir(parents=True, exist_ok=True)

    counts = {"images": 0, "masks": 0}
    if not image_source.exists() or not mask_source.exists():
        return counts

    for image_path in iter_images(image_source):
        mask_candidates = [
            mask_source / image_path.name,
            mask_source / f"{image_path.stem}.png",
            mask_source / f"{image_path.stem}.jpg",
            mask_source / f"{image_path.stem}.jpeg",
        ]
        mask_path = next((candidate for candidate in mask_candidates if candidate.exists()), None)
        if mask_path is None:
            continue

        destination_image = image_destination / f"{image_path.stem}.png"
        destination_mask = mask_destination / f"{image_path.stem}.png"
        resize_and_save_image(image_path, destination_image, size, grayscale=True)
        resize_and_save_image(mask_path, destination_mask, size, grayscale=True)
        counts["images"] += 1
        counts["masks"] += 1

    return counts


def preprocess_gan_images(raw_root: Path, processed_root: Path, size: tuple[int, int]):
    source_root = raw_root / "gan_images"
    destination_root = processed_root / "gan_images"
    destination_root.mkdir(parents=True, exist_ok=True)

    count = 0
    if not source_root.exists():
        return count

    for image_path in iter_images(source_root):
        destination_path = destination_root / f"{image_path.stem}.png"
        resize_and_save_image(image_path, destination_path, size, grayscale=True)
        count += 1

    return count


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess raw MRI data into the folder layout used by the challenge project.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Directory containing raw challenge data.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"), help="Directory to write processed data into.")
    parser.add_argument("--detection-size", type=int, nargs=2, default=(224, 224), help="Detection image size.")
    parser.add_argument("--classification-size", type=int, nargs=2, default=(224, 224), help="Classification image size.")
    parser.add_argument("--segmentation-size", type=int, nargs=2, default=(256, 256), help="Segmentation image size.")
    parser.add_argument("--gan-size", type=int, nargs=2, default=(64, 64), help="GAN image size.")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_structure(args.processed_dir)

    detection_counts = preprocess_detection(args.raw_dir, args.processed_dir, tuple(args.detection_size))
    classification_counts = preprocess_classification(args.raw_dir, args.processed_dir, tuple(args.classification_size))
    segmentation_counts = preprocess_segmentation(args.raw_dir, args.processed_dir, tuple(args.segmentation_size))
    gan_count = preprocess_gan_images(args.raw_dir, args.processed_dir, tuple(args.gan_size))

    print("Preprocessing complete.")
    print(f"Detection counts: {detection_counts}")
    print(f"Classification counts: {classification_counts}")
    print(f"Segmentation counts: {segmentation_counts}")
    print(f"GAN image count: {gan_count}")


if __name__ == "__main__":
    main()
