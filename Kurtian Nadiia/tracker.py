from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(slots=True)
class Hypothesis:
    bbox: List[float]
    vel: np.ndarray
    score: float
    kind: str
    age: int = 0
    tracker: Optional[Any] = None
    size_ref: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0], dtype=np.float32)
    )
    missed: int = 0
    visible_streak: int = 1
    hidden: bool = False
    template_gray: Optional[np.ndarray] = None
    template_edge: Optional[np.ndarray] = None
    appearance_hist: Optional[np.ndarray] = None
    template_mean: float = 0.0
    template_std: float = 0.0
    edge_density: float = 0.0
    quality: float = 0.0


class AdvancedMHT:
    """
    Robust MHT-style tracker, tuned for CHASE CAMERA mode.
    ULTIMATE SURVIVAL CALIBRATION: Includes 'Last Stand' fallback
    to guarantee track survival during extended occlusions.
    """

    def __init__(
            self,
            frame: np.ndarray,
            bbox: Sequence[float],
            *,
            max_hypotheses: int = 12,
            max_velocity: float = 30.0,
            iou_threshold: float = 0.35,
            scale_tolerance: Tuple[float, float] = (0.70, 1.40),
            velocity_alpha_visual: float = 0.20,
            velocity_alpha_motion: float = 0.05,
            size_ema: float = 0.12,
            score_decay: float = 0.985,
            visual_bonus: float = 0.16,
            reacquire_bonus: float = 0.35,
            occlusion_penalty: float = -0.015,  # ПОМ'ЯКШЕНО штраф за перебування в тіні
            branch_penalty: float = -0.04,
            min_score: float = 0.015,
            max_missed: int = 90,  # КРИТИЧНО: Збільшено вдвічі (чекаємо до 3 секунд)
            appearance_alpha: float = 0.18,
            template_update_alpha: float = 0.01,
            template_min_score: float = 0.38,
            reacquire_score_threshold: float = 0.60,
            tracker_quality_threshold: float = 0.45,
            search_expand: float = 1.6,
    ) -> None:
        self._validate_frame(frame)
        self.frame_height, self.frame_width = frame.shape[:2]

        self.max_hypotheses = int(max_hypotheses)
        self.max_velocity = float(max_velocity)
        self.iou_threshold = float(iou_threshold)
        self.scale_min, self.scale_max = map(float, scale_tolerance)
        self.velocity_alpha_visual = float(velocity_alpha_visual)
        self.velocity_alpha_motion = float(velocity_alpha_motion)
        self.size_ema = float(size_ema)
        self.score_decay = float(score_decay)
        self.visual_bonus = float(visual_bonus)
        self.reacquire_bonus = float(reacquire_bonus)
        self.occlusion_penalty = float(occlusion_penalty)
        self.branch_penalty = float(branch_penalty)
        self.min_score = float(min_score)
        self.max_missed = int(max_missed)
        self.appearance_alpha = float(appearance_alpha)
        self.template_update_alpha = float(template_update_alpha)
        self.template_min_score = float(template_min_score)
        self.reacquire_score_threshold = float(reacquire_score_threshold)
        self.tracker_quality_threshold = float(tracker_quality_threshold)
        self.search_expand = float(search_expand)

        self.search_directions: Tuple[Tuple[float, float, str], ...] = (
            (0.0, 0.0, "CENTER"),
            (-0.5, 0.0, "LEFT"),
            (0.5, 0.0, "RIGHT"),
            (0.0, -0.5, "UP"),
            (0.0, 0.5, "DOWN"),
        )

        initial_bbox = self._clamp_bbox(bbox)
        initial_tracker = self._init_tracker(frame, initial_bbox)
        initial_template = self._extract_template(frame, initial_bbox)
        initial_edge = self._extract_edge_template(frame, initial_bbox)
        initial_hist = self._extract_histogram(frame, initial_bbox)
        initial_mean, initial_std, initial_edge_density = self._template_stats(initial_template, initial_edge)

        self.hypotheses: List[Hypothesis] = [
            Hypothesis(
                bbox=initial_bbox,
                vel=np.zeros(2, dtype=np.float32),
                score=1.0,
                kind="ROOT",
                age=0,
                tracker=initial_tracker,
                size_ref=np.array(initial_bbox[2:4], dtype=np.float32),
                missed=0,
                visible_streak=1,
                hidden=False,
                template_gray=initial_template,
                template_edge=initial_edge,
                appearance_hist=initial_hist,
                template_mean=initial_mean,
                template_std=initial_std,
                edge_density=initial_edge_density,
                quality=1.0,
            )
        ]

    def update(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        self._validate_frame(frame)
        self.frame_height, self.frame_width = frame.shape[:2]

        candidates: List[Hypothesis] = []

        for hypothesis in self.hypotheses:
            parent_age = hypothesis.age + 1
            pred_box = self._apply_velocity(hypothesis.bbox, hypothesis.vel)
            measurements = self._get_visual_measurements(frame, hypothesis, pred_box)

            if measurements:
                for measurement in measurements:
                    score_delta = self.visual_bonus + measurement["quality"] * 0.22
                    if hypothesis.missed > 0:
                        score_delta += self.reacquire_bonus

                    should_update = measurement["quality"] > 0.88
                    if should_update:
                        current_template = self._extract_template(frame, measurement["bbox"])
                        current_edge = self._extract_edge_template(frame, measurement["bbox"])
                        updated_template = self._blend_template(hypothesis.template_gray, current_template)
                        updated_edge = self._blend_template(hypothesis.template_edge, current_edge)
                        updated_hist = self._blend_histogram(hypothesis.appearance_hist,
                                                             self._extract_histogram(frame, measurement["bbox"]))
                    else:
                        updated_template = hypothesis.template_gray
                        updated_edge = hypothesis.template_edge
                        updated_hist = hypothesis.appearance_hist

                    tracker_instance = None
                    if measurement["quality"] >= self.reacquire_score_threshold:
                        tracker_instance = self._init_tracker(frame, measurement["bbox"])

                    updated_mean, updated_std, updated_edge_density = self._template_stats(updated_template,
                                                                                           updated_edge)
                    candidates.append(
                        self._create_child(
                            parent=hypothesis,
                            new_bbox=measurement["bbox"],
                            score_delta=score_delta,
                            kind=measurement["kind"],
                            age=parent_age,
                            alpha=self.velocity_alpha_visual,
                            size_ref=self._update_size_ref(hypothesis.size_ref, measurement["bbox"]),
                            update_velocity=True,
                            tracker_instance=tracker_instance,
                            missed=0,
                            hidden=False,
                            visible_streak=hypothesis.visible_streak + 1,
                            template_gray=updated_template,
                            template_edge=updated_edge,
                            appearance_hist=updated_hist,
                            template_mean=updated_mean,
                            template_std=updated_std,
                            edge_density=updated_edge_density,
                            quality=measurement["quality"],
                        )
                    )
            else:
                # ПОЛІПШЕНО: Штраф за оклюзію тепер зростає набагато повільніше (0.003 замість 0.012)
                propagated = self._create_child(
                    parent=hypothesis,
                    new_bbox=pred_box,
                    score_delta=self.occlusion_penalty - 0.003 * hypothesis.missed,
                    kind="OCCLUDED",
                    age=parent_age,
                    alpha=self.velocity_alpha_motion,
                    size_ref=hypothesis.size_ref.copy(),
                    update_velocity=False,
                    tracker_instance=None,
                    missed=hypothesis.missed + 1,
                    hidden=True,
                    visible_streak=0,
                    template_gray=hypothesis.template_gray,
                    template_edge=hypothesis.template_edge,
                    appearance_hist=hypothesis.appearance_hist,
                    template_mean=hypothesis.template_mean,
                    template_std=hypothesis.template_std,
                    edge_density=hypothesis.edge_density,
                    quality=max(0.0, hypothesis.quality * 0.9),
                )
                candidates.append(propagated)

                radius = self._dynamic_search_radius(hypothesis, pred_box)
                for dx_norm, dy_norm, name in self.search_directions[1:]:
                    branch_box = pred_box.copy()
                    branch_box[0] += dx_norm * radius
                    branch_box[1] += dy_norm * radius
                    candidates.append(
                        self._create_child(
                            parent=hypothesis,
                            new_bbox=self._clamp_bbox(branch_box),
                            score_delta=self.occlusion_penalty + self.branch_penalty - 0.005 * hypothesis.missed,
                            kind=f"OCCLUDED_{name}",
                            age=parent_age,
                            alpha=self.velocity_alpha_motion,
                            size_ref=hypothesis.size_ref.copy(),
                            update_velocity=False,
                            tracker_instance=None,
                            missed=hypothesis.missed + 1,
                            hidden=True,
                            visible_streak=0,
                            template_gray=hypothesis.template_gray,
                            template_edge=hypothesis.template_edge,
                            appearance_hist=hypothesis.appearance_hist,
                            template_mean=hypothesis.template_mean,
                            template_std=hypothesis.template_std,
                            edge_density=hypothesis.edge_density,
                            quality=max(0.0, hypothesis.quality * 0.85),
                        )
                    )

        self.hypotheses = self._prune_and_rank(candidates, frame)
        if not self.hypotheses:
            return None

        best = self.hypotheses[0]
        return {
            "bbox": best.bbox,
            "vel": best.vel.tolist(),
            "score": float(best.score),
            "type": best.kind,
            "age": best.age,
            "missed": best.missed,
            "hidden": bool(best.hidden),
            "quality": float(best.quality),
        }

    def _create_child(
            self,
            *,
            parent: Hypothesis,
            new_bbox: Sequence[float],
            score_delta: float,
            kind: str,
            age: int,
            alpha: float,
            size_ref: np.ndarray,
            update_velocity: bool,
            tracker_instance: Optional[Any],
            missed: int,
            hidden: bool,
            visible_streak: int,
            template_gray: Optional[np.ndarray],
            template_edge: Optional[np.ndarray],
            appearance_hist: Optional[np.ndarray],
            template_mean: float,
            template_std: float,
            edge_density: float,
            quality: float,
    ) -> Hypothesis:
        new_bbox = self._clamp_bbox(new_bbox)

        if update_velocity:
            raw_velocity = np.array(
                [new_bbox[0] - parent.bbox[0], new_bbox[1] - parent.bbox[1]],
                dtype=np.float32,
            )
            raw_velocity = np.clip(raw_velocity, -self.max_velocity, self.max_velocity)
            filtered_velocity = alpha * raw_velocity + (1.0 - alpha) * parent.vel
        else:
            filtered_velocity = parent.vel * 0.1

        hidden_penalty = 0.0 if not hidden else 0.005 * min(missed, self.max_missed)  # Пом'якшено
        streak_bonus = min(visible_streak, 10) * 0.003

        new_score = float(
            np.clip(
                parent.score * self.score_decay + score_delta - hidden_penalty + streak_bonus,
                0.0,  # Дозволяємо падати до 0 під час розрахунку, щоб потім відфільтрувати
                2.5,
            )
        )

        return Hypothesis(
            bbox=new_bbox,
            vel=filtered_velocity.astype(np.float32),
            score=new_score,
            kind=kind,
            age=age,
            tracker=tracker_instance,
            size_ref=size_ref.astype(np.float32),
            missed=missed,
            visible_streak=visible_streak,
            hidden=hidden,
            template_gray=template_gray,
            template_edge=template_edge,
            appearance_hist=appearance_hist,
            template_mean=float(template_mean),
            template_std=float(template_std),
            edge_density=float(edge_density),
            quality=float(np.clip(quality, 0.0, 1.0)),
        )

    def _prune_and_rank(self, candidates: List[Hypothesis], frame: np.ndarray) -> List[Hypothesis]:
        if not candidates:
            return []

        valid_candidates = [c for c in candidates if c.missed <= self.max_missed and c.score >= self.min_score]

        if not valid_candidates:
            alive_candidates = [c for c in candidates if c.missed <= self.max_missed]
            if alive_candidates:
                alive_candidates.sort(key=lambda c: c.score, reverse=True)
                hero = alive_candidates[0]
                hero.score = self.min_score + 0.001  # Даємо кисень для виживання
                valid_candidates = [hero]

        if not valid_candidates:
            return []

        valid_candidates.sort(key=self._ranking_key, reverse=True)

        survivors: List[Hypothesis] = []
        for candidate in valid_candidates:
            if len(survivors) >= self.max_hypotheses:
                break

            duplicate = False
            for survivor in survivors:
                if self._calculate_iou(candidate.bbox, survivor.bbox) > self.iou_threshold:
                    if survivor.hidden == candidate.hidden:
                        duplicate = True
                        break
            if duplicate:
                continue

            if candidate.tracker is None and not candidate.hidden and candidate.quality >= self.reacquire_score_threshold:
                candidate.tracker = self._init_tracker(frame, candidate.bbox)

            survivors.append(candidate)

        if not survivors:
            return []

        best_score = max(h.score for h in survivors)
        norm = max(best_score, self.min_score)
        for hypothesis in survivors:
            hypothesis.score = float(np.clip(hypothesis.score / norm, self.min_score, 1.0))

        survivors.sort(key=self._ranking_key, reverse=True)
        return survivors

    def _ranking_key(self, hypothesis: Hypothesis) -> Tuple[float, float, float, float, int]:
        visible_bias = 0.08 if not hypothesis.hidden else 0.0
        reacquire_bias = 0.05 if hypothesis.kind.startswith("TM_") else 0.0
        return (
            hypothesis.score + visible_bias + reacquire_bias,
            hypothesis.quality,
            -float(hypothesis.missed),
            float(hypothesis.visible_streak),
            -hypothesis.age,
        )

    def _whiteness_score(self, frame: np.ndarray, bbox: Sequence[float]) -> float:
        patch = self._extract_patch(frame, bbox, inner_ratio=0.50)
        if patch is None or patch.size == 0:
            return 0.0

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 70])
        upper_white = np.array([180, 90, 255])

        mask = cv2.inRange(hsv, lower_white, upper_white)
        white_pixels = cv2.countNonZero(mask)
        total_pixels = patch.shape[0] * patch.shape[1] + 1e-6

        white_ratio = white_pixels / total_pixels

        if white_ratio < 0.005:
            return 0.0

        return float(np.clip(white_ratio * 30.0, 0.0, 1.0))

    def _get_visual_measurements(
            self,
            frame: np.ndarray,
            hypothesis: Hypothesis,
            pred_box: Sequence[float],
    ) -> List[Dict[str, Any]]:
        measurements: List[Dict[str, Any]] = []

        tracker_measurement = self._tracker_measurement(frame, hypothesis)
        if tracker_measurement is not None:
            measurements.append(tracker_measurement)

        template_measurements = self._template_measurements(frame, hypothesis, pred_box)
        measurements.extend(template_measurements)

        if not measurements:
            return []

        measurements.sort(key=lambda item: item["quality"], reverse=True)

        filtered: List[Dict[str, Any]] = []
        for measurement in measurements:
            is_duplicate = any(
                self._calculate_iou(measurement["bbox"], kept["bbox"]) > 0.45
                for kept in filtered
            )
            if is_duplicate:
                continue
            filtered.append(measurement)
            if len(filtered) >= 3:
                break

        return filtered

    def _tracker_measurement(
            self,
            frame: np.ndarray,
            hypothesis: Hypothesis,
    ) -> Optional[Dict[str, Any]]:
        if hypothesis.tracker is None:
            return None

        try:
            success, tracked_box = hypothesis.tracker.update(frame)
        except cv2.error:
            return None

        if not success:
            return None

        tracked_box = self._clamp_bbox(tracked_box)
        if self._is_significant_change(hypothesis.bbox, tracked_box, hypothesis.vel, multiplier=1.1):
            return None

        scale_ok = self._is_scale_consistent(hypothesis.size_ref, tracked_box)
        if not scale_ok:
            return None

        appearance_score = self._appearance_score(frame, tracked_box, hypothesis)
        if appearance_score < 0.15:
            return None

        motion_score = self._motion_score(hypothesis, tracked_box)
        quality = 0.62 * appearance_score + 0.38 * motion_score

        if quality < self.tracker_quality_threshold:
            return None

        return {
            "bbox": tracked_box,
            "quality": float(np.clip(quality, 0.0, 1.0)),
            "kind": "TRACKER",
        }

    def _template_measurements(
            self,
            frame: np.ndarray,
            hypothesis: Hypothesis,
            pred_box: Sequence[float],
    ) -> List[Dict[str, Any]]:
        template = hypothesis.template_gray
        template_edge = hypothesis.template_edge
        if template is None or template.size == 0:
            return []

        gray = self._to_gray(frame)
        edges = self._to_edges(gray)
        search_roi = self._get_search_region(pred_box, hypothesis.missed, hypothesis.vel, hypothesis.size_ref)
        x0, y0, rw, rh = search_roi

        if rw < max(20, template.shape[1]) or rh < max(20, template.shape[0]):
            return []

        search_patch = gray[y0: y0 + rh, x0: x0 + rw]
        edge_patch = edges[y0: y0 + rh, x0: x0 + rw]
        if search_patch.size == 0 or edge_patch.size == 0:
            return []

        scales = self._candidate_scales(hypothesis.missed)
        candidates: List[Dict[str, Any]] = []
        seen: List[List[float]] = []

        for scale in scales:
            target_w = max(16, int(round(float(hypothesis.size_ref[0]) * scale)))
            target_h = max(16, int(round(float(hypothesis.size_ref[1]) * scale)))
            if target_w >= rw or target_h >= rh:
                continue

            templ = cv2.resize(template, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            templ_edge = None
            if template_edge is not None and template_edge.size > 0:
                templ_edge = cv2.resize(template_edge, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            if templ.shape[0] < 8 or templ.shape[1] < 8:
                continue

            try:
                gray_result = cv2.matchTemplate(search_patch, templ, cv2.TM_CCOEFF_NORMED)
                if templ_edge is not None and edge_patch.shape[0] >= templ_edge.shape[0] and edge_patch.shape[1] >= \
                        templ_edge.shape[1]:
                    edge_result = cv2.matchTemplate(edge_patch, templ_edge, cv2.TM_CCOEFF_NORMED)
                    result = 0.55 * gray_result + 0.45 * edge_result
                else:
                    result = gray_result
            except cv2.error:
                continue

            if result.size == 0:
                continue

            top_candidates = self._topk_from_heatmap(result, k=3)
            for peak_score, peak_x, peak_y in top_candidates:
                bbox = self._clamp_bbox([x0 + peak_x, y0 + peak_y, target_w, target_h])
                if not self._is_scale_consistent(hypothesis.size_ref, bbox):
                    continue
                if self._is_significant_change(hypothesis.bbox, bbox, hypothesis.vel, multiplier=1.5):
                    continue

                appearance_score = self._appearance_score(frame, bbox, hypothesis)

                if appearance_score < 0.15:
                    continue

                motion_score = self._motion_score(hypothesis, bbox)
                quality = 0.45 * float(peak_score) + 0.35 * appearance_score + 0.20 * motion_score

                if hypothesis.missed > 0:
                    quality += min(0.12, 0.02 * hypothesis.missed)
                quality = float(np.clip(quality, 0.0, 1.0))

                if quality < self.template_min_score:
                    continue

                duplicate = any(self._calculate_iou(bbox, prev_box) > 0.45 for prev_box in seen)
                if duplicate:
                    continue
                seen.append(bbox)
                candidates.append(
                    {
                        "bbox": bbox,
                        "quality": quality,
                        "kind": f"TM_x{scale:.2f}",
                    }
                )

        candidates.sort(key=lambda item: item["quality"], reverse=True)
        return candidates[:3]

    def _candidate_scales(self, missed: int) -> Tuple[float, ...]:
        if missed == 0:
            return (0.95, 1.00, 1.05)
        if missed < 8:
            return (0.90, 1.00, 1.10)
        return (0.75, 0.85, 0.95, 1.00, 1.10, 1.25)

    def _get_search_region(
            self,
            pred_box: Sequence[float],
            missed: int,
            vel: np.ndarray,
            size_ref: np.ndarray,
    ) -> Tuple[int, int, int, int]:
        x, y, w, h = map(float, pred_box)
        speed = float(np.linalg.norm(vel))

        expand = min(self.search_expand + 0.1 * missed + speed / 60.0, 3.5)

        region_w = max(32.0, float(size_ref[0]) * expand)
        region_h = max(32.0, float(size_ref[1]) * expand)

        cx = x + w * 0.5
        cy = y + h * 0.5
        left = int(max(0.0, cx - region_w * 0.5))
        top = int(max(0.0, cy - region_h * 0.5))
        right = int(min(float(self.frame_width), cx + region_w * 0.5))
        bottom = int(min(float(self.frame_height), cy + region_h * 0.5))

        width = max(1, right - left)
        height = max(1, bottom - top)
        return left, top, width, height

    def _dynamic_search_radius(self, hypothesis: Hypothesis, pred_box: Sequence[float]) -> float:
        speed = float(np.linalg.norm(hypothesis.vel))
        base = max(float(pred_box[2]), float(pred_box[3])) * 0.3
        return min(60.0, base + 2.0 * hypothesis.missed + 0.15 * speed)

    def _motion_score(self, hypothesis: Hypothesis, bbox: Sequence[float]) -> float:
        predicted = self._apply_velocity(hypothesis.bbox, hypothesis.vel)
        dx = abs(float(bbox[0]) - float(predicted[0]))
        dy = abs(float(bbox[1]) - float(predicted[1]))
        dw = abs(float(bbox[2]) - float(hypothesis.size_ref[0]))
        dh = abs(float(bbox[3]) - float(hypothesis.size_ref[1]))

        denom_x = max(float(hypothesis.size_ref[0]) * 1.6, abs(float(hypothesis.vel[0])) * 2.0 + 8.0)
        denom_y = max(float(hypothesis.size_ref[1]) * 1.6, abs(float(hypothesis.vel[1])) * 2.0 + 8.0)
        denom_w = max(float(hypothesis.size_ref[0]) * 0.8, 8.0)
        denom_h = max(float(hypothesis.size_ref[1]) * 0.8, 8.0)

        norm = (dx / denom_x) + (dy / denom_y) + 0.4 * (dw / denom_w + dh / denom_h)
        return float(np.clip(1.0 - 0.35 * norm, 0.0, 1.0))

    def _appearance_score(
            self,
            frame: np.ndarray,
            bbox: Sequence[float],
            hypothesis: Hypothesis,
    ) -> float:
        white_score = self._whiteness_score(frame, bbox)
        if white_score == 0.0:
            return 0.0

        template_score = 0.5
        if hypothesis.template_gray is not None:
            current_template = self._extract_template(frame, bbox)
            if current_template is not None and current_template.size > 0:
                ref = hypothesis.template_gray
                aligned = cv2.resize(current_template, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LINEAR)
                try:
                    match = cv2.matchTemplate(aligned, ref, cv2.TM_CCOEFF_NORMED)
                    if match.size > 0:
                        template_score = float(np.clip(match.max(), 0.0, 1.0))
                except cv2.error:
                    template_score = 0.5

        base_score = float(np.clip(template_score, 0.0, 1.0))
        return base_score * white_score

    def _is_scale_consistent(self, size_ref: np.ndarray, bbox: Sequence[float]) -> bool:
        w_ratio = float(bbox[2]) / max(float(size_ref[0]), 1e-6)
        h_ratio = float(bbox[3]) / max(float(size_ref[1]), 1e-6)
        return self.scale_min <= w_ratio <= self.scale_max and self.scale_min <= h_ratio <= self.scale_max

    def _is_significant_change(
            self,
            old_bbox: Sequence[float],
            new_bbox: Sequence[float],
            velocity: np.ndarray,
            *,
            multiplier: float = 1.3,
    ) -> bool:
        old_x, old_y, old_w, old_h = map(float, old_bbox)
        new_x, new_y, new_w, new_h = map(float, new_bbox)

        expected_jump_x = abs(float(velocity[0]))
        expected_jump_y = abs(float(velocity[1]))

        pos_change_x = abs(new_x - old_x)
        pos_change_y = abs(new_y - old_y)
        size_change = abs(new_w - old_w) + abs(new_h - old_h)

        max_allowed_dx = max(multiplier * old_w, expected_jump_x * 1.5 + 10.0)
        max_allowed_dy = max(multiplier * old_h, expected_jump_y * 1.5 + 10.0)

        if pos_change_x > max_allowed_dx or pos_change_y > max_allowed_dy:
            return True
        if size_change > max(old_w, old_h) * 0.7:
            return True

        return False

    def _apply_velocity(self, bbox: Sequence[float], vel: np.ndarray) -> List[float]:
        return self._clamp_bbox(
            [
                float(bbox[0]) + float(vel[0]),
                float(bbox[1]) + float(vel[1]),
                float(bbox[2]),
                float(bbox[3]),
            ]
        )

    def _update_size_ref(self, previous_size_ref: np.ndarray, bbox: Sequence[float]) -> np.ndarray:
        current_size = np.array([bbox[2], bbox[3]], dtype=np.float32)
        return (1.0 - self.size_ema) * previous_size_ref + self.size_ema * current_size

    def _blend_histogram(
            self,
            previous_hist: Optional[np.ndarray],
            current_hist: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if current_hist is None:
            return previous_hist
        if previous_hist is None:
            return current_hist
        blended = (1.0 - self.appearance_alpha) * previous_hist + self.appearance_alpha * current_hist
        norm = float(blended.sum())
        if norm > 1e-6:
            blended = blended / norm
        return blended.astype(np.float32)

    def _blend_template(
            self,
            previous_template: Optional[np.ndarray],
            current_template: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if current_template is None:
            return previous_template
        if previous_template is None:
            return current_template

        h, w = previous_template.shape[:2]
        aligned = cv2.resize(current_template, (w, h), interpolation=cv2.INTER_LINEAR)
        blended = cv2.addWeighted(
            previous_template.astype(np.float32),
            1.0 - self.template_update_alpha,
            aligned.astype(np.float32),
            self.template_update_alpha,
            0.0,
        )
        return np.clip(blended, 0.0, 255.0).astype(np.uint8)

    def _extract_template(self, frame: np.ndarray, bbox: Sequence[float]) -> Optional[np.ndarray]:
        patch = self._extract_patch(frame, bbox, inner_ratio=0.70)
        if patch is None or patch.size == 0:
            return None
        return self._to_gray(patch)

    def _extract_histogram(self, frame: np.ndarray, bbox: Sequence[float]) -> Optional[np.ndarray]:
        patch = self._extract_patch(frame, bbox, inner_ratio=0.60)
        if patch is None or patch.size == 0:
            return None
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [1, 2], None, [16, 16], [0, 256, 0, 256])
        hist = hist.astype(np.float32)
        total = float(hist.sum())
        if total <= 1e-6:
            return None
        hist /= total
        return hist

    def _extract_edge_template(self, frame: np.ndarray, bbox: Sequence[float]) -> Optional[np.ndarray]:
        templ = self._extract_template(frame, bbox)
        if templ is None or templ.size == 0:
            return None
        blurred = cv2.GaussianBlur(templ, (3, 3), 0)
        return cv2.Canny(blurred, 40, 130)

    @staticmethod
    def _template_stats(template_gray: Optional[np.ndarray], template_edge: Optional[np.ndarray]) -> Tuple[
        float, float, float]:
        if template_gray is None or template_gray.size == 0:
            return 0.0, 0.0, 0.0
        mean = float(np.mean(template_gray))
        std = float(np.std(template_gray))
        edge_density = 0.0
        if template_edge is not None and template_edge.size > 0:
            edge_density = float(np.mean(template_edge > 0))
        return mean, std, edge_density

    def _extract_patch(self, frame: np.ndarray, bbox: Sequence[float], inner_ratio: float = 1.0) -> Optional[
        np.ndarray]:
        x, y, w, h = map(float, self._clamp_bbox(bbox))
        if inner_ratio < 1.0:
            new_w = w * inner_ratio
            new_h = h * inner_ratio
            x += (w - new_w) * 0.5
            y += (h - new_h) * 0.5
            w, h = new_w, new_h
        x, y, w, h = map(int, self._clamp_bbox([x, y, w, h]))

        if w <= 0 or h <= 0 or x >= self.frame_width or y >= self.frame_height:
            return None

        patch = frame[y: y + h, x: x + w]
        if patch.size == 0:
            return None
        return patch

    def _init_tracker(self, frame: np.ndarray, bbox: Sequence[float]) -> Optional[Any]:
        tracker = self._make_csrt_tracker()
        if tracker is None:
            return None
        try:
            tracker.init(frame, tuple(map(int, self._clamp_bbox(bbox))))
        except cv2.error:
            return None
        return tracker

    @staticmethod
    def _make_csrt_tracker() -> Optional[Any]:
        constructors = [
            lambda: cv2.TrackerCSRT_create(),
            lambda: cv2.legacy.TrackerCSRT_create(),
        ]
        for constructor in constructors:
            try:
                return constructor()
            except (AttributeError, cv2.error):
                continue
        return None

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _to_edges(gray: np.ndarray) -> np.ndarray:
        if gray.ndim != 2:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.Canny(blurred, 40, 130)

    @staticmethod
    def _topk_from_heatmap(heatmap: np.ndarray, k: int = 2) -> List[Tuple[float, int, int]]:
        if heatmap.size == 0:
            return []
        flat = heatmap.reshape(-1)
        count = min(k, flat.size)
        indices = np.argpartition(flat, -count)[-count:]
        indices = indices[np.argsort(flat[indices])[::-1]]
        width = heatmap.shape[1]
        peaks: List[Tuple[float, int, int]] = []
        for idx in indices:
            y = int(idx // width)
            x = int(idx % width)
            peaks.append((float(flat[idx]), x, y))
        return peaks

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty numpy array")

    def _clamp_bbox(self, bbox: Sequence[float]) -> List[float]:
        if len(bbox) != 4:
            raise ValueError("bbox must contain exactly 4 values: [x, y, w, h]")

        x, y, w, h = map(float, bbox)
        w = max(1.0, w)
        h = max(1.0, h)
        x = max(0.0, min(x, float(self.frame_width - 1)))
        y = max(0.0, min(y, float(self.frame_height - 1)))
        w = max(1.0, min(w, float(self.frame_width) - x))
        h = max(1.0, min(h, float(self.frame_height) - y))
        return [x, y, w, h]

    @staticmethod
    def _calculate_iou(b1: Sequence[float], b2: Sequence[float]) -> float:
        x_left = max(float(b1[0]), float(b2[0]))
        y_top = max(float(b1[1]), float(b2[1]))
        x_right = min(float(b1[0]) + float(b1[2]), float(b2[0]) + float(b2[2]))
        y_bottom = min(float(b1[1]) + float(b1[3]), float(b2[1]) + float(b2[3]))

        inter_w = max(0.0, x_right - x_left)
        inter_h = max(0.0, y_bottom - y_top)
        intersection = inter_w * inter_h
        if intersection <= 0.0:
            return 0.0

        area1 = max(1.0, float(b1[2]) * float(b1[3]))
        area2 = max(1.0, float(b2[2]) * float(b2[3]))
        union = area1 + area2 - intersection
        return float(intersection / max(union, 1e-6))