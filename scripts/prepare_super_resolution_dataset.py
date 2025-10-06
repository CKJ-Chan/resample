"""Utility for preparing super-resolution datasets.

The script normalises arbitrary image folders into the directory layout
expected by ``sample_condition.py`` for the super-resolution task. It
converts the images to PNG files, optionally enforces grayscale intensity
profiles (useful for black-and-white medical imagery), and resizes them to
a square crop so that the high-resolution images match the measurement
configuration used in the ReSample repository and paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageOps


VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")


def iter_image_paths(directory: Path, extensions: Sequence[str]) -> Iterable[Path]:
    for ext in extensions:
        yield from directory.rglob(f"*{ext}")


def _to_rgb(img: Image.Image, force_grayscale: bool) -> Image.Image:
    """Return an RGB image, optionally replicating grayscale channels."""

    if force_grayscale:
        gray = ImageOps.grayscale(img)
        return Image.merge("RGB", (gray, gray, gray))

    if img.mode == "RGB":
        return img
    if img.mode == "RGBA":
        return img.convert("RGB")

    gray = ImageOps.grayscale(img)
    return Image.merge("RGB", (gray, gray, gray))


def prepare_split(
    src: Path,
    dst: Path,
    image_size: int,
    skip_existing: bool,
    force_grayscale: bool,
) -> list[Path]:
    image_paths = sorted({path.resolve() for path in iter_image_paths(src, VALID_EXTENSIONS)})
    if not image_paths:
        raise FileNotFoundError(f"No images with extensions {VALID_EXTENSIONS} were found in {src}.")

    dst.mkdir(parents=True, exist_ok=True)

    written_paths: list[Path] = []

    for path in image_paths:
        relative_name = path.stem + ".png"
        out_path = dst / relative_name

        if skip_existing and out_path.exists():
            written_paths.append(out_path.resolve())
            continue

        with Image.open(path) as img:
            img = _to_rgb(img, force_grayscale=force_grayscale)
            img = ImageOps.fit(img, (image_size, image_size), method=Image.Resampling.LANCZOS)
            img.save(out_path)

        written_paths.append(out_path.resolve())

    return written_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a super-resolution dataset")
    parser.add_argument("--train-dir", type=Path, required=True,
                        help="Directory that contains the high-resolution training images.")
    parser.add_argument("--test-dir", type=Path, required=False,
                        help="Directory that contains the high-resolution test images.")
    parser.add_argument("--output-root", type=Path, default=Path("data/super_resolution"),
                        help="Root directory where the processed dataset will be stored. Splits will be "
                             "created under this directory.")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Target resolution for the high-resolution images.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip processing images that already exist in the destination directory.")
    parser.add_argument("--grayscale", action="store_true",
                        help="Replicate grayscale intensity into three channels so that downstream models "
                             "(which expect RGB tensors) still receive monochrome content. Recommended for "
                             "medical imagery.")
    parser.add_argument("--write-manifests", action="store_true",
                        help="Export text files with absolute paths to the processed images. These manifests "
                             "can be fed to the PyTorch Lightning training pipeline (see README).")
    return parser


def main(cli_args: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(cli_args)

    splits = {"train": args.train_dir}
    if args.test_dir is not None:
        splits["test"] = args.test_dir

    manifest_entries: dict[str, list[Path]] = {}

    for split, src_dir in splits.items():
        if not src_dir.exists():
            raise FileNotFoundError(f"The provided directory for the '{split}' split does not exist: {src_dir}")

        dst_dir = args.output_root / split
        print(f"Processing {split} split: {src_dir} -> {dst_dir}")
        manifest_entries[split] = prepare_split(
            src_dir,
            dst_dir,
            args.image_size,
            args.skip_existing,
            force_grayscale=args.grayscale,
        )

    if args.write_manifests:
        manifest_root = args.output_root
        manifest_root.mkdir(parents=True, exist_ok=True)

        for split, paths in manifest_entries.items():
            manifest_path = manifest_root / f"{split}_images.txt"
            with manifest_path.open("w", encoding="utf-8") as handle:
                handle.write("\n".join(str(path) for path in paths))
            print(f"Wrote manifest with {len(paths)} entries to {manifest_path}")


if __name__ == "__main__":
    main()
