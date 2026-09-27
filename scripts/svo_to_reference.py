#!/usr/bin/env python3
"""
Convert Stereolabs SVO / SVO2 file to:
1. Standard MP4 (H.264 / mp4v) for visualization and inspection.
2. JEPA reference HDF5 episode (observations/images/camera_front) for World Model planning.
Supports trimming start/end frames and downsampling FPS.
"""

import sys
import argparse
from pathlib import Path
import cv2
import h5py
import numpy as np
import pyzed.sl as sl


def convert_svo(
    svo_path: str,
    output_h5: str = None,
    output_mp4: str = None,
    start_frame: int = 0,
    end_frame: int = -1,
    subsample_step: int = 1,
):
    svo_file = Path(svo_path).resolve()
    if not svo_file.exists():
        raise FileNotFoundError(f"SVO file not found: {svo_file}")

    print(f"[SVO] Opening SVO: {svo_file.name}")
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_file))
    init_params.svo_real_time_mode = False
    init_params.depth_mode = sl.DEPTH_MODE.NONE  # Fast RGB only

    cam = sl.Camera()
    status = cam.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open SVO: {status}")

    nb_total_frames = cam.get_svo_number_of_frames()
    cam_info = cam.get_camera_information()
    fps = cam_info.camera_configuration.fps or 30.0
    res = cam_info.camera_configuration.resolution
    print(f"[SVO] Info: {nb_total_frames} frames | {fps} FPS | {res.width}x{res.height}")

    eff_start = max(0, start_frame)
    eff_end = nb_total_frames if (end_frame <= 0 or end_frame > nb_total_frames) else end_frame
    print(f"[SVO] Extracting frames [{eff_start} -> {eff_end}] (step: {subsample_step})...")

    # Set read position
    cam.set_svo_position(eff_start)

    image_mat = sl.Mat()
    frames_rgb = []

    # Optional MP4 writer
    mp4_writer = None
    out_fps = fps / subsample_step
    if output_mp4:
        Path(output_mp4).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        mp4_writer = cv2.VideoWriter(str(output_mp4), fourcc, out_fps, (res.width, res.height))

    curr_frame_idx = eff_start
    while curr_frame_idx < eff_end:
        if cam.grab() == sl.ERROR_CODE.SUCCESS:
            if (curr_frame_idx - eff_start) % subsample_step == 0:
                cam.retrieve_image(image_mat, sl.VIEW.LEFT)
                bgra = image_mat.get_data()
                bgr = bgra[:, :, :3]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames_rgb.append(rgb)

                if mp4_writer is not None:
                    mp4_writer.write(bgr)

            curr_frame_idx += 1
            if len(frames_rgb) % 100 == 0:
                print(f"  Processed frame {curr_frame_idx}/{eff_end} (saved: {len(frames_rgb)})...")
        else:
            break

    cam.close()
    if mp4_writer is not None:
        mp4_writer.release()
        print(f"[SVO] Saved MP4: {output_mp4} ({len(frames_rgb)} frames @ {out_fps:.1f} FPS)")

    if output_h5 and len(frames_rgb) > 0:
        images_np = np.stack(frames_rgb, axis=0)
        out_h5 = Path(output_h5)
        out_h5.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(str(out_h5), "w") as f:
            obs = f.create_group("observations")
            img_grp = obs.create_group("images")
            ds = img_grp.create_dataset(
                "camera_front",
                data=images_np,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            ds.attrs["color_space"] = "RGB"
            ds.attrs["source_fps"] = out_fps
            ds.attrs["source_svo"] = str(svo_file)
            ds.attrs["start_frame"] = eff_start
            ds.attrs["end_frame"] = eff_end

        print(f"[SVO] Saved HDF5: {out_h5} ({len(images_np)} RGB frames, {images_np.shape[1]}x{images_np.shape[2]})")


def main():
    parser = argparse.ArgumentParser(description="Convert ZED SVO/SVO2 to MP4 and JEPA HDF5")
    parser.add_argument("svo_path", type=str, help="Path to input .svo / .svo2 file")
    parser.add_argument("--output_mp4", type=str, default=None, help="Output MP4 video path")
    parser.add_argument("--output_h5", type=str, default=None, help="Output HDF5 dataset path")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index")
    parser.add_argument("--end_frame", type=int, default=-1, help="End frame index (-1 for end of file)")
    parser.add_argument("--step", type=int, default=1, help="Frame subsample step (e.g. 2 for 15 FPS from 30 FPS)")
    args = parser.parse_args()

    convert_svo(
        svo_path=args.svo_path,
        output_h5=args.output_h5,
        output_mp4=args.output_mp4,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        subsample_step=args.step,
    )


if __name__ == "__main__":
    main()
