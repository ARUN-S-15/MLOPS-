import cv2
import os
import argparse
from pathlib import Path


def extract_frames_uniform(video_path: Path, out_dir: Path, target_frames: int, resize: tuple | None = None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        print(f"Could not determine frame count for {video_path}; falling back to sequential read")

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    if total == 0 or total <= target_frames:
        # read sequentially and save all (or available) frames
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if resize:
                frame = cv2.resize(frame, resize)
            fname = out_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(fname), frame)
            idx += 1
            saved += 1
    else:
        # sample uniformly to get exactly target_frames
        import math

        stride = total / target_frames
        for i in range(target_frames):
            frame_idx = int(i * stride)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            if resize:
                frame = cv2.resize(frame, resize)
            fname = out_dir / f"frame_{i:06d}.jpg"
            cv2.imwrite(str(fname), frame)
            saved += 1

    cap.release()
    return saved


def main():
    p = argparse.ArgumentParser(description="Extract a target number of frames per video by uniform sampling.")
    p.add_argument("--input", "-i", default="DATASET/videodata", help="Input folder with videos")
    p.add_argument("--output", "-o", default="DATASET/frames", help="Output root for extracted frames")
    p.add_argument("--target", "-t", type=int, default=900, help="Target frames per video (recommended 800-1000)")
    p.add_argument("--resize", nargs=2, type=int, metavar=("W", "H"), help="Optional resize (width height)")
    args = p.parse_args()

    input_dir = Path(args.input)
    output_root = Path(args.output)
    target = args.target
    resize = tuple(args.resize) if args.resize else None

    VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".mpeg", ".flv"}

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return

    videos = [p for p in input_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    if not videos:
        print(f"No videos found in {input_dir}")
        return

    for vid in videos:
        out_dir = output_root / vid.stem
        print(f"Processing {vid} -> {out_dir} (target {target})")
        saved = extract_frames_uniform(vid, out_dir, target, resize)
        print(f"Saved {saved} frames for {vid.name}")


if __name__ == "__main__":
    main()
