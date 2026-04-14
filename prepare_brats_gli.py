from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


MODALITY_SUFFIX = {
    "t1c": "-t1c.nii.gz",
    "t1n": "-t1n.nii.gz",
    "t2f": "-t2f.nii.gz",
    "t2w": "-t2w.nii.gz",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert BraTS-GLI NIfTI data into processed PNG slices for this project.")
    parser.add_argument("--source", type=Path, default=Path("data"), help="Folder containing BraTS-GLI-* case directories.")
    parser.add_argument("--target", type=Path, default=Path("data/processed"), help="Processed output root.")
    parser.add_argument("--modality", choices=sorted(MODALITY_SUFFIX.keys()), default="t2f", help="MRI modality to export as 2D slices.")
    parser.add_argument("--min-tumour-pixels", type=int, default=20, help="Minimum tumour-mask pixels to call a slice tumour-positive.")
    parser.add_argument("--max-slices-per-case", type=int, default=60, help="Maximum slices to export per case after filtering.")
    parser.add_argument("--image-size", type=int, nargs=2, default=(256, 256), help="Output image size (H W).")
    return parser.parse_args()


def normalize_to_uint8(slice_array: np.ndarray) -> np.ndarray:
    arr = slice_array.astype(np.float32)
    p1, p99 = np.percentile(arr, [1, 99])
    arr = np.clip(arr, p1, p99)
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    return (arr * 255).astype(np.uint8)


def ensure_dirs(root: Path):
    for path in [
        root / "detection" / "normal",
        root / "detection" / "tumour",
        root / "classification" / "glioma",
        root / "segmentation" / "images",
        root / "segmentation" / "masks",
        root / "gan_images",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def save_png(array: np.ndarray, out_path: Path, image_size: tuple[int, int]):
    image = Image.fromarray(array)
    image = image.resize((image_size[1], image_size[0]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def case_paths(case_dir: Path, modality: str):
    case_id = case_dir.name
    image_path = case_dir / f"{case_id}{MODALITY_SUFFIX[modality]}"
    seg_path = case_dir / f"{case_id}-seg.nii.gz"
    return image_path, seg_path


def choose_slices(mask_volume: np.ndarray, max_slices: int, min_tumour_pixels: int):
    depth = mask_volume.shape[-1]
    tumour_indices = []
    normal_indices = []

    for idx in range(depth):
        pixels = int((mask_volume[..., idx] > 0).sum())
        if pixels >= min_tumour_pixels:
            tumour_indices.append(idx)
        else:
            normal_indices.append(idx)

    tumour_keep = tumour_indices[: max_slices // 2] if tumour_indices else []
    normal_keep = normal_indices[: max_slices - len(tumour_keep)] if normal_indices else []
    return sorted(set(tumour_keep + normal_keep))


def process_case(case_dir: Path, target: Path, modality: str, image_size: tuple[int, int], min_tumour_pixels: int, max_slices: int):
    image_path, seg_path = case_paths(case_dir, modality)
    if not image_path.exists() or not seg_path.exists():
        return {"saved": 0, "tumour": 0, "normal": 0}

    image_volume = nib.load(str(image_path)).get_fdata()
    seg_volume = nib.load(str(seg_path)).get_fdata()
    selected = choose_slices(seg_volume, max_slices=max_slices, min_tumour_pixels=min_tumour_pixels)

    saved = tumour = normal = 0
    for idx in selected:
        image_slice = image_volume[..., idx]
        seg_slice = (seg_volume[..., idx] > 0).astype(np.uint8) * 255
        tumour_pixels = int((seg_slice > 0).sum())
        label = "tumour" if tumour_pixels >= min_tumour_pixels else "normal"

        image_uint8 = normalize_to_uint8(image_slice)
        stem = f"{case_dir.name}_z{idx:03d}"

        save_png(image_uint8, target / "segmentation" / "images" / f"{stem}.png", image_size)
        save_png(seg_slice, target / "segmentation" / "masks" / f"{stem}.png", image_size)
        save_png(image_uint8, target / "detection" / label / f"{stem}.png", image_size)
        save_png(image_uint8, target / "gan_images" / f"{stem}.png", image_size)

        if label == "tumour":
            save_png(image_uint8, target / "classification" / "glioma" / f"{stem}.png", image_size)
            tumour += 1
        else:
            normal += 1

        saved += 1

    return {"saved": saved, "tumour": tumour, "normal": normal}


def main():
    args = parse_args()
    ensure_dirs(args.target)

    case_dirs = [path for path in sorted(args.source.iterdir()) if path.is_dir() and path.name.startswith("BraTS-GLI-")]
    total_saved = total_tumour = total_normal = 0

    for case_dir in case_dirs:
        stats = process_case(
            case_dir=case_dir,
            target=args.target,
            modality=args.modality,
            image_size=tuple(args.image_size),
            min_tumour_pixels=args.min_tumour_pixels,
            max_slices=args.max_slices_per_case,
        )
        total_saved += stats["saved"]
        total_tumour += stats["tumour"]
        total_normal += stats["normal"]

    print(f"Cases processed: {len(case_dirs)}")
    print(f"Total slices exported: {total_saved}")
    print(f"Detection tumour slices: {total_tumour}")
    print(f"Detection normal slices: {total_normal}")
    print(f"Output root: {args.target}")


if __name__ == "__main__":
    main()
