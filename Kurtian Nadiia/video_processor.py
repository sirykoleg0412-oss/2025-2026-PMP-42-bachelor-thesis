from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from tracker import AdvancedMHT


class VideoProcessor:
    def __init__(
        self,
        video_path: str,
        *,
        window_name: str = "Improved MHT Tracker",
        output_path: Optional[str] = None,
        display: bool = True,
    ):
        self.video_path = video_path
        self.window_name = window_name
        self.output_path = output_path
        self.display = display

    def run(self) -> None:
        cap = None
        writer = None

        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"Unable to open video file '{self.video_path}'")
                return

            ret, frame = cap.read()
            if not ret or frame is None:
                print("Unable to read the first frame")
                return

            roi = self._select_roi(frame)
            if roi is None:
                print("ROI selection cancelled.")
                return

            mht = AdvancedMHT(frame, list(roi))

            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            if self.output_path:
                writer = self._create_writer(self.output_path, fps, frame.shape[1], frame.shape[0])

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Video has ended.")
                    break

                if frame is None or frame.size == 0:
                    print("Empty or invalid frame detected!")
                    break

                try:
                    best_h_dict = mht.update(frame)
                    if best_h_dict:
                        self._draw_hypotheses(frame, mht.hypotheses, best_h_dict)
                    else:
                        cv2.putText(
                            frame,
                            "Track lost",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 255),
                            2,
                        )
                except Exception as exc:
                    print(f"Error during update: {exc}")
                    traceback.print_exc()
                    break

                if writer is not None:
                    writer.write(frame)

                if self.display:
                    cv2.imshow(self.window_name, frame)
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord("q"):
                        print("Processing has been stopped.")
                        break
                else:
                    cv2.waitKey(1)

        except Exception as exc:
            print(f"Critical Error: {exc}")
            traceback.print_exc()

        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()

    def _select_roi(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        try:
            roi = cv2.selectROI(
                self.window_name,
                frame,
                fromCenter=False,
                showCrosshair=True,
            )
        except Exception as exc:
            print(f"Error while selecting ROI: {exc}")
            return None

        if roi == (0, 0, 0, 0):
            return None

        print(f"Selected ROI: {roi}")
        return tuple(map(int, roi))

    def _create_writer(self, output_path: str, fps: float, width: int, height: int):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Unable to create output video '{output_path}'")
        return writer

    def _draw_hypotheses(
        self,
        frame: np.ndarray,
        hypotheses: List[Any],
        best_h: Dict[str, Any],
    ) -> None:
        for hyp in hypotheses:
            x, y, w, h = map(int, hyp.bbox)
            color = (255, 180, 0) if hyp.hidden else (0, 220, 0)
            thickness = 1 if hyp is not hypotheses[0] else 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

        x, y, w, h = map(int, best_h["bbox"])
        best_color = (0, 165, 255) if best_h.get("hidden") else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), best_color, 3)

        info_lines = [
            f"Active hypotheses: {len(hypotheses)}",
            f"Best: {best_h['type']}",
            f"Score: {best_h['score']:.2f}",
            f"Quality: {best_h.get('quality', 0.0):.2f}",
            f"Missed: {best_h.get('missed', 0)}",
        ]

        if best_h.get("hidden"):
            info_lines.append("State: OCCLUDED / PREDICTED")
        else:
            info_lines.append("State: VISIBLE")

        origin_x, origin_y = 15, 30
        for idx, line in enumerate(info_lines):
            cv2.putText(
                frame,
                line,
                (origin_x, origin_y + idx * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (20, 20, 20),
                4,
            )
            cv2.putText(
                frame,
                line,
                (origin_x, origin_y + idx * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
            )

        label = f"{best_h['type']} | miss={best_h.get('missed', 0)}"
        cv2.putText(
            frame,
            label,
            (x, max(20, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
