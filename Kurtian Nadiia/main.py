from __future__ import annotations

import argparse
from pathlib import Path

from video_processor import VideoProcessor

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Improved MHT tracker with occlusion handling")
    parser.add_argument(
        "--video",
        default="drone_flight.mp4",
        help="Path to input video",

    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to save annotated output video",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without showing the OpenCV window",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    video_path = str(Path(args.video))
    output_path = str(Path(args.output)) if args.output else None

    processor = VideoProcessor(
        video_path,
        output_path=output_path,
        display=not args.no_display,
    )
    processor.run()


if __name__ == "__main__":
    main()
