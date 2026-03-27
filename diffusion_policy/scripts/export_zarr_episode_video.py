if __name__ == "__main__":
    import pathlib
    import sys

    ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(ROOT_DIR))

import argparse
import subprocess

import numpy as np
import zarr


def get_episode_bounds(episode_ends: np.ndarray, episode_idx: int) -> tuple[int, int]:
    if episode_idx < 0 or episode_idx >= len(episode_ends):
        raise IndexError(
            f"episode_idx={episode_idx} out of range for {len(episode_ends)} episodes"
        )
    start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    end = int(episode_ends[episode_idx])
    return start, end


def chw_rgb_to_hwc(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3:
        raise ValueError(f"Expected 3D frame, got shape {frame.shape}")
    if frame.shape[0] != 3:
        raise ValueError(f"Expected RGB channels-first frame, got shape {frame.shape}")
    return np.transpose(frame, (1, 2, 0))


def export_video(
    frames: np.ndarray,
    output_path: pathlib.Path,
    fps: int,
    codec: str,
    crf: int,
) -> None:
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError(f"Expected frames with shape (T, 3, H, W), got {frames.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _, _, height, width = frames.shape
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        codec,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for frame in frames:
            process.stdin.write(chw_rgb_to_hwc(frame).astype(np.uint8, copy=False).tobytes())
        process.stdin.close()
        return_code = process.wait()
    except Exception:
        process.kill()
        process.wait()
        raise
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")


def main():
    parser = argparse.ArgumentParser(
        description="Export one episode from a zarr dataset to mp4."
    )
    parser.add_argument(
        "--input",
        default="data/datasets/stack_blocks_two",
        help="Path to the zarr dataset directory.",
    )
    parser.add_argument(
        "--camera-key",
        default="head_camera",
        help="Camera dataset key under data/.",
    )
    parser.add_argument(
        "--episode-idx",
        type=int,
        default=0,
        help="0-based episode index to export.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Output video fps.",
    )
    parser.add_argument(
        "--codec",
        default="libx264",
        help="ffmpeg video codec.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="ffmpeg CRF quality value. Lower means higher quality.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output mp4 path. Defaults to <input>/episode_<idx>_<camera>.mp4",
    )
    args = parser.parse_args()

    input_path = pathlib.Path(args.input).expanduser().resolve()
    root = zarr.open(str(input_path), mode="r")

    episode_ends = root["meta"]["episode_ends"][:]
    start, end = get_episode_bounds(episode_ends, args.episode_idx)
    frames = root["data"][args.camera_key][start:end]

    if args.output is None:
        output_path = input_path / f"episode_{args.episode_idx:03d}_{args.camera_key}.mp4"
    else:
        output_path = pathlib.Path(args.output).expanduser().resolve()

    export_video(
        frames=frames,
        output_path=output_path,
        fps=args.fps,
        codec=args.codec,
        crf=args.crf,
    )
    print(
        f"saved episode {args.episode_idx} ({end - start} frames) "
        f"from {args.camera_key} to {output_path}"
    )


if __name__ == "__main__":
    main()
