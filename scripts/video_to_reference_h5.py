#!/usr/bin/env python3
"""Convert a visual demonstration video into a Demo-JEPA reference episode."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an RGB video to observations/images/camera_front HDF5."
    )
    parser.add_argument("input_video", type=Path)
    parser.add_argument("output_h5", type=Path)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap for a short smoke-test episode.",
    )
    args = parser.parse_args()

    if not args.input_video.is_file():
        raise FileNotFoundError(f"Input video not found: {args.input_video}")

    capture = cv2.VideoCapture(str(args.input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.input_video}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while args.max_frames is None or len(frames) < args.max_frames:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    capture.release()

    if len(frames) < 2:
        raise ValueError("The reference episode requires at least two decodable frames.")

    images = np.stack(frames, axis=0)
    args.output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output_h5, "w") as handle:
        observations = handle.create_group("observations")
        image_group = observations.create_group("images")
        dataset = image_group.create_dataset(
            "camera_front",
            data=images,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        dataset.attrs["color_space"] = "RGB"
        dataset.attrs["source_fps"] = fps
        dataset.attrs["source_video"] = str(args.input_video.resolve())

    print(
        f"Wrote {args.output_h5}: {len(images)} RGB frames at {fps:g} FPS "
        f"({images.shape[1]}x{images.shape[2]})."
    )


if __name__ == "__main__":
    main()
