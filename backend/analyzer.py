"""
Skating Video Analyzer (Enhanced Visualization Version)
-------------------------------------------------------
- YOLOX + RTMPose.
- Scale-Invariant Contact Detection.
- Enhanced Drawing: Colors + Feet connections (Mimics original rtmlib).
"""

from __future__ import annotations
import cv2
import logging
import math
import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# --- 1. Import Models Only ---
try:
    from rtmlib.tools import RTMPose, YOLOX
except ImportError:
    RTMPose = YOLOX = None


# --- 2. Enhanced Custom Drawing Function ---
def draw_skeleton(img, kpts, scores, kpt_thr=0.3, radius=4, line_width=2):
    """
    Draws a colorful skeleton including feet (Halpe-26 topology).
    Mimics the style of rtmlib/mmpose.
    """
    kpts = np.array(kpts)
    scores = np.array(scores)

    if kpts.ndim == 3:
        kpts = kpts[0]
    if scores.ndim == 2:
        scores = scores[0]

    # Color Palette (B, G, R)
    c_red = (0, 0, 255)
    c_orange = (0, 165, 255)
    c_green = (0, 255, 0)
    c_cyan = (255, 255, 0)
    c_blue = (255, 0, 0)
    c_purple = (255, 0, 255)
    c_white = (255, 255, 255)

    # (Link_Start, Link_End, Color)
    skeleton_config = [
        # Body (Center) - Purple/Blue
        (5, 6, c_purple),  # Shoulders
        (11, 12, c_purple),  # Hips
        (5, 11, c_blue),  # L-Torso
        (6, 12, c_blue),  # R-Torso
        # Head - White
        (0, 1, c_white),
        (0, 2, c_white),
        (1, 3, c_white),
        (2, 4, c_white),
        # Right Arm/Leg (Red/Orange)
        (6, 8, c_red),
        (8, 10, c_red),  # Arm
        (12, 14, c_orange),
        (14, 16, c_orange),  # Leg
        # Left Arm/Leg (Green/Cyan)
        (5, 7, c_green),
        (7, 9, c_green),  # Arm
        (11, 13, c_cyan),
        (13, 15, c_cyan),  # Leg
        # Feet (Halpe-26 specific) - Crucial for Skating
        # Right Foot (Orange)
        (16, 21, c_orange),  # Ankle -> BigToe
        (16, 25, c_orange),  # Ankle -> Heel
        (21, 25, c_orange),  # Heel -> BigToe (Triangle)
        # Left Foot (Cyan)
        (15, 20, c_cyan),  # Ankle -> BigToe
        (15, 24, c_cyan),  # Ankle -> Heel
        (20, 24, c_cyan),  # Heel -> BigToe (Triangle)
    ]

    # 1. Draw Links (Bones)
    for i, j, color in skeleton_config:
        # Check bounds (some models might not output 26 points)
        if i < len(scores) and j < len(scores):
            if scores[i] > kpt_thr and scores[j] > kpt_thr:
                p1 = (int(kpts[i][0]), int(kpts[i][1]))
                p2 = (int(kpts[j][0]), int(kpts[j][1]))
                cv2.line(img, p1, p2, color, line_width, cv2.LINE_AA)

    # 2. Draw Keypoints (Joints)
    for i, p in enumerate(kpts):
        if i < len(scores) and scores[i] > kpt_thr:
            # Color logic for joints
            if i in [0, 1, 2, 3, 4]:
                c = c_white
            elif i in [6, 8, 10, 12, 14, 16, 21, 23, 25]:
                c = c_red  # Right side
            elif i in [5, 7, 9, 11, 13, 15, 20, 22, 24]:
                c = c_green  # Left side
            else:
                c = c_purple

            cv2.circle(img, (int(p[0]), int(p[1])), radius, c, -1)
            cv2.circle(
                img, (int(p[0]), int(p[1])), radius, (0, 0, 0), 1
            )  # Black outline

    return img


def draw_bbox(img, boxes, **kw):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        # Green box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img


# -----------------------------
# Keypoint Constants
# -----------------------------
KEYPOINT_DICT = {
    "NOSE": 0,
    "L_EYE": 1,
    "R_EYE": 2,
    "L_EAR": 3,
    "R_EAR": 4,
    "L_SHOULDER": 5,
    "R_SHOULDER": 6,
    "L_ELBOW": 7,
    "R_ELBOW": 8,
    "L_WRIST": 9,
    "R_WRIST": 10,
    "L_HIP": 11,
    "R_HIP": 12,
    "L_KNEE": 13,
    "R_KNEE": 14,
    "L_ANKLE": 15,
    "R_ANKLE": 16,
    "HEAD": 17,
    "NECK": 18,
    "HIP_CENTER": 19,
    "L_BIG_TOE": 20,
    "R_BIG_TOE": 21,
    "L_SMALL_TOE": 22,
    "R_SMALL_TOE": 23,
    "L_HEEL": 24,
    "R_HEEL": 25,
}

L_SHOULDER, R_SHOULDER = KEYPOINT_DICT["L_SHOULDER"], KEYPOINT_DICT["R_SHOULDER"]
L_ELBOW, R_ELBOW = KEYPOINT_DICT["L_ELBOW"], KEYPOINT_DICT["R_ELBOW"]
L_WRIST, R_WRIST = KEYPOINT_DICT["L_WRIST"], KEYPOINT_DICT["R_WRIST"]
L_HIP, R_HIP = KEYPOINT_DICT["L_HIP"], KEYPOINT_DICT["R_HIP"]
L_KNEE, R_KNEE = KEYPOINT_DICT["L_KNEE"], KEYPOINT_DICT["R_KNEE"]
L_ANKLE, R_ANKLE = KEYPOINT_DICT["L_ANKLE"], KEYPOINT_DICT["R_ANKLE"]
L_BIG_TOE, R_BIG_TOE = KEYPOINT_DICT["L_BIG_TOE"], KEYPOINT_DICT["R_BIG_TOE"]
L_HEEL, R_HEEL = KEYPOINT_DICT["L_HEEL"], KEYPOINT_DICT["R_HEEL"]
NOSE = KEYPOINT_DICT["NOSE"]
L_EAR, R_EAR = KEYPOINT_DICT["L_EAR"], KEYPOINT_DICT["R_EAR"]


# -----------------------------
# Math Helpers
# -----------------------------
def xyc(kp):
    if kp is None:
        return None
    a = np.asarray(kp)
    return (
        (float(a[0]), float(a[1]), float(a[2]))
        if a.size >= 3
        else (float(a[0]), float(a[1]), 1.0)
    )


def midpoint(p1, p2, min_conf=0.0):
    a, b = xyc(p1), xyc(p2)
    if a is None or b is None or a[2] < min_conf or b[2] < min_conf:
        return None
    return np.array(
        [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), min(a[2], b[2])], dtype=np.float32
    )


def vec(a, b):
    if a is None or b is None:
        return None
    aa, bb = np.asarray(a)[:2], np.asarray(b)[:2]
    return bb - aa


def angle_between(v1, v2):
    if v1 is None or v2 is None:
        return None
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, np.dot(v1, v2) / (n1 * n2)))))


def iou_xyxy(a, b):
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    inter_w = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    inter = inter_w * inter_h
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    return float(inter / (area_a + area_b - inter + 1e-6))


# -----------------------------
# Classes
# -----------------------------
@dataclass
class FootIndices:
    ankle: int
    heel: int
    big_toe: int
    small_toe: Optional[int] = None


class LRTemporalResolver:
    def __init__(
        self, conf_thr=0.3, min_swap_improve=12.0, persist_frames=2, ema_alpha=0.6
    ):
        self.conf_thr = float(conf_thr)
        self.min_swap_improve = float(min_swap_improve)
        self.persist_frames = int(persist_frames)
        self.ema_alpha = float(ema_alpha)
        self.prev_kps = None
        self.prev_lens = {
            "L": {"thigh": None, "shank": None},
            "R": {"thigh": None, "shank": None},
        }
        self._swap_streak = 0

    def _p(self, kps, scores, idx):
        if idx is None or scores[idx] < self.conf_thr:
            return None
        return np.array(kps[idx, :2], dtype=float)

    def _dist(self, a, b):
        return (
            float(np.linalg.norm(a - b)) if (a is not None and b is not None) else 0.0
        )

    def _get_lengths(self, assign):
        lens = {"L": {}, "R": {}}
        for side in ("L", "R"):
            lens[side]["thigh"] = self._dist(assign[side]["hip"], assign[side]["knee"])
            lens[side]["shank"] = self._dist(
                assign[side]["knee"], assign[side]["ankle"]
            )
        return lens

    def _anatomy_cost(self, cur_lens):
        cost = 0.0
        for side in ("L", "R"):
            for part in ("thigh", "shank"):
                curr = cur_lens[side][part]
                prev = self.prev_lens[side][part]
                if curr > 0 and prev is not None and prev > 0:
                    ratio = abs(curr - prev) / prev
                    if ratio > 0.2:
                        cost += 100.0 * ratio
        return cost

    def _pack(self, kps, scores, order=("L", "R")):
        sides = {}
        for side, A, B, T, H in (
            (order[0], L_ANKLE, L_KNEE, L_BIG_TOE, L_HEEL),
            (order[1], R_ANKLE, R_KNEE, R_BIG_TOE, R_HEEL),
        ):
            sides[side] = {
                "ankle": self._p(kps, scores, A),
                "knee": self._p(kps, scores, B),
                "toe": self._p(kps, scores, T),
                "heel": self._p(kps, scores, H),
                "hip": self._p(kps, scores, (L_HIP if side == "L" else R_HIP)),
            }
        return sides

    def resolve(self, kps, scores):
        cur_id = self._pack(kps, scores, order=("L", "R"))
        cur_sw = self._pack(kps, scores, order=("R", "L"))

        if self.prev_kps is None:
            self.prev_kps = cur_id
            self.prev_lens = self._get_lengths(cur_id)
            return kps, scores

        def move_cost(c, p):
            cost = 0.0
            for side in ("L", "R"):
                for k in ("ankle", "knee", "hip"):
                    cost += self._dist(c[side][k], p[side][k])
            return cost

        total_id = move_cost(cur_id, self.prev_kps) + self._anatomy_cost(
            self._get_lengths(cur_id)
        )
        total_sw = move_cost(cur_sw, self.prev_kps) + self._anatomy_cost(
            self._get_lengths(cur_sw)
        )

        if total_sw + self.min_swap_improve < total_id:
            self._swap_streak += 1
        else:
            self._swap_streak = 0

        do_swap = self._swap_streak >= self.persist_frames
        chosen = cur_sw if do_swap else cur_id
        chosen_lens = self._get_lengths(chosen)

        alpha = self.ema_alpha
        for side in ("L", "R"):
            for k in ("ankle", "knee", "hip"):
                c, p = chosen[side][k], self.prev_kps[side][k]
                if c is not None:
                    self.prev_kps[side][k] = (
                        c if p is None else (1 - alpha) * p + alpha * c
                    )
            for part in ("thigh", "shank"):
                c, p = chosen_lens[side][part], self.prev_lens[side][part]
                if c > 0:
                    self.prev_lens[side][part] = c if p is None else 0.9 * p + 0.1 * c

        if not do_swap:
            return kps, scores

        K, S = np.asarray(kps).copy(), np.asarray(scores).copy()
        # Constants now defined!
        swap_pairs = [
            (L_ANKLE, R_ANKLE),
            (L_KNEE, R_KNEE),
            (L_HEEL, R_HEEL),
            (L_BIG_TOE, R_BIG_TOE),
            (L_HIP, R_HIP),
            (L_SHOULDER, R_SHOULDER),
            (L_ELBOW, R_ELBOW),
            (L_WRIST, R_WRIST),
            (L_EAR, R_EAR),
        ]
        for a, b in swap_pairs:
            K[[a, b]], S[[a, b]] = K[[b, a]], S[[b, a]]
        return K, S


class SkateContactAnalyzer:
    def __init__(
        self,
        left_idx,
        right_idx,
        ground_tolerance_ratio=0.15,
        pitch_threshold_ratio=0.25,
        min_conf=0.3,
        **kw,
    ):
        self.left_idx = left_idx
        self.right_idx = right_idx
        self.min_conf = min_conf
        self.ground_tolerance = ground_tolerance_ratio
        self.pitch_threshold = pitch_threshold_ratio

    def _get_coords(self, kps, idx_obj):
        a_idx, h_idx, t_idx = idx_obj.ankle, idx_obj.heel, idx_obj.big_toe
        if max(a_idx, h_idx, t_idx) >= len(kps):
            return None, None, None
        a, h, t = kps[a_idx], kps[h_idx], kps[t_idx]
        if len(a) > 2 and (
            a[2] < self.min_conf or h[2] < self.min_conf or t[2] < self.min_conf
        ):
            return None, None, None
        return a[:2], h[:2], t[:2]

    def _dist(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def update(self, kps, scores, t):
        La, Lh, Lt = self._get_coords(kps, self.left_idx)
        Ra, Rh, Rt = self._get_coords(kps, self.right_idx)

        def get_shank(k, a):
            return (
                self._dist(kps[k, :2], kps[a, :2])
                if (scores[k] >= self.min_conf and scores[a] >= self.min_conf)
                else 0.0
            )

        scale_unit = max(get_shank(L_KNEE, L_ANKLE), get_shank(R_KNEE, R_ANKLE), 1.0)

        cands = []
        if Lh is not None:
            cands.extend([Lh[1], Lt[1]])
        if Rh is not None:
            cands.extend([Rh[1], Rt[1]])
        ground_y = max(cands) if cands else 0.0

        def classify(a, h, t):
            if a is None or not cands:
                return "Unknown"
            if (ground_y - max(h[1], t[1])) >= (self.ground_tolerance * scale_unit):
                return "AIR"

            foot_len = max(self._dist(h, t), 1.0)
            norm_pitch = (t[1] - h[1]) / foot_len

            if norm_pitch > self.pitch_threshold:
                return "TOE"
            elif norm_pitch < -self.pitch_threshold:
                return "HEEL"
            return "FLAT"

        L_state = classify(La, Lh, Lt)
        R_state = classify(Ra, Rh, Rt)

        def calc_ang(s):
            idx = self.left_idx if s == "left" else self.right_idx
            k = L_KNEE if s == "left" else R_KNEE
            if scores[k] < self.min_conf or scores[idx.ankle] < self.min_conf:
                return {"angle": None}
            return {
                "angle": round(
                    abs(
                        angle_between(
                            kps[k, :2] - kps[idx.ankle, :2],
                            kps[idx.big_toe, :2] - kps[idx.ankle, :2],
                        )
                    ),
                    1,
                )
            }

        return {
            "left": {"state": L_state},
            "right": {"state": R_state},
            "wheeling": {"left": calc_ang("left"), "right": calc_ang("right")},
        }


def extract_color_histogram(frame, box, bins=16):
    """
    Extract a normalized color histogram from a bounding box region.
    Used for appearance-based re-identification.
    """
    x1, y1, x2, y2 = map(int, box[:4])
    H, W = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    roi_small = cv2.resize(roi, (32, 64))
    hsv = cv2.cvtColor(roi_small, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist = np.concatenate([hist_h.flatten(), hist_s.flatten()])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.astype(np.float32)


def histogram_similarity(hist1, hist2):
    """Compare two histograms using correlation. Returns 0-1 similarity score."""
    if hist1 is None or hist2 is None:
        return 0.0
    if hist1.shape != hist2.shape:
        return 0.0
    return max(0.0, cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))


class BoxTracker:
    def __init__(self, iou_thr=0.2, dist_thr=400):
        self.iou_thr = iou_thr
        self.dist_thr = dist_thr
        self.tracks = {}
        self.next_id = 0
        self.target_id = None
        self.target_appearance = None
        self.target_box = None

    def _centroid(self, box):
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    def _dist(self, c1, c2):
        return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

    def _aspect_ratio(self, box):
        w = box[2] - box[0]
        h = box[3] - box[1]
        return w / max(h, 1.0)

    def set_target(self, target_id, appearance_hist=None, box=None):
        """Mark a track as the primary target to follow."""
        self.target_id = target_id
        self.target_appearance = appearance_hist
        self.target_box = box.copy() if box is not None else None
        if target_id in self.tracks:
            self.tracks[target_id]["is_target"] = True
            if appearance_hist is not None:
                self.tracks[target_id]["appearance"] = appearance_hist

    def update(self, boxes: List[np.ndarray], frame_shape, appearances: List = None):
        assigned_ids = [-1] * len(boxes)
        used_tracks = set()

        for i, box in enumerate(boxes):
            if assigned_ids[i] != -1:
                continue
            best_id, best_score = -1, -1

            for tid, track in self.tracks.items():
                if tid in used_tracks:
                    continue
                iou_score = iou_xyxy(track["box"], box)
                if iou_score > best_score:
                    best_score, best_id = iou_score, tid

            if best_score >= self.iou_thr:
                assigned_ids[i] = best_id
                used_tracks.add(best_id)
                self.tracks[best_id].update({"box": box, "age": 0})
                if appearances and i < len(appearances) and appearances[i] is not None:
                    old_app = self.tracks[best_id].get("appearance")
                    if old_app is not None:
                        self.tracks[best_id]["appearance"] = (
                            0.7 * old_app + 0.3 * appearances[i]
                        )
                    else:
                        self.tracks[best_id]["appearance"] = appearances[i]
                if best_id == self.target_id:
                    self.target_box = box.copy()

        for i, box in enumerate(boxes):
            if assigned_ids[i] != -1:
                continue
            best_id, best_dist = -1, float("inf")
            c_curr = self._centroid(box)
            aspect_curr = self._aspect_ratio(box)

            for tid, track in self.tracks.items():
                if tid in used_tracks:
                    continue
                c_prev = self._centroid(track["box"])
                dist = self._dist(c_curr, c_prev)
                aspect_prev = self._aspect_ratio(track["box"])
                aspect_diff = abs(aspect_curr - aspect_prev)
                combined_score = dist + 50 * aspect_diff
                if combined_score < best_dist:
                    best_dist, best_id = combined_score, tid

            max_dist_threshold = self.dist_thr
            if best_id == self.target_id:
                max_dist_threshold = self.dist_thr * 1.5

            if best_dist < max_dist_threshold:
                assigned_ids[i] = best_id
                used_tracks.add(best_id)
                self.tracks[best_id].update({"box": box, "age": 0})
                if appearances and i < len(appearances) and appearances[i] is not None:
                    old_app = self.tracks[best_id].get("appearance")
                    if old_app is not None:
                        self.tracks[best_id]["appearance"] = (
                            0.7 * old_app + 0.3 * appearances[i]
                        )
                    else:
                        self.tracks[best_id]["appearance"] = appearances[i]
                if best_id == self.target_id:
                    self.target_box = box.copy()

        for i in range(len(boxes)):
            if assigned_ids[i] == -1:
                self.tracks[self.next_id] = {
                    "box": boxes[i],
                    "age": 0,
                    "is_target": False,
                }
                if appearances and i < len(appearances) and appearances[i] is not None:
                    self.tracks[self.next_id]["appearance"] = appearances[i]
                assigned_ids[i] = self.next_id
                self.next_id += 1

        for tid in list(self.tracks.keys()):
            if tid not in used_tracks:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > 30:
                    if tid == self.target_id:
                        self.target_id = None
                        self.target_appearance = None
                        self.target_box = None
                    del self.tracks[tid]

        return assigned_ids

    def find_best_target_match(self, boxes, appearances, frame_shape):
        """Find the best match for the target using appearance similarity."""
        if self.target_appearance is None or not boxes:
            return None, -1

        best_idx, best_score = -1, 0.0
        target_aspect = None
        if self.target_box is not None:
            target_aspect = self._aspect_ratio(self.target_box)

        for i, (box, app) in enumerate(zip(boxes, appearances)):
            if app is None:
                continue
            sim = histogram_similarity(self.target_appearance, app)
            if target_aspect is not None:
                aspect_diff = abs(self._aspect_ratio(box) - target_aspect)
                sim = sim * (1.0 - 0.3 * min(aspect_diff, 1.0))
            if sim > best_score:
                best_score = sim
                best_idx = i

        return best_idx, best_score


def posture_metrics(kps, scores, conf_thr=0.3):
    ms = midpoint(kps[L_SHOULDER], kps[R_SHOULDER], conf_thr)
    mh = midpoint(kps[L_HIP], kps[R_HIP], conf_thr)
    torso = (
        angle_between(vec(mh, ms), [0, -1])
        if (ms is not None and mh is not None)
        else None
    )
    return torso, None, None, None


def knee_angle(kps, scores, side, conf_thr=0.3):
    h, k, a = (L_HIP, L_KNEE, L_ANKLE) if side == "left" else (R_HIP, R_KNEE, R_ANKLE)
    if min(scores[h], scores[k], scores[a]) < conf_thr:
        return None
    return 180.0 - angle_between(vec(kps[h], kps[k]), vec(kps[k], kps[a]))


class MetricLogger:
    def __init__(self, path):
        self.f = open(path, "w", newline="") if path else None
        if self.f:
            import csv

            self.w = csv.DictWriter(
                self.f,
                fieldnames=[
                    "frame",
                    "time_s",
                    "subject",
                    "wheelL_ang",
                    "wheelR_ang",
                    "torso_deg",
                    "kneeL_deg",
                    "kneeR_deg",
                ],
            )
            self.w.writeheader()

    def write_row(self, row):
        if self.f:
            self.w.writerow(row)

    def close(self):
        if self.f:
            self.f.close()


# -----------------------------
# Main Logic Class
# -----------------------------
class SkatingVideoAnalyzer:
    def __init__(
        self,
        yolo_model,
        rtm_model,
        process_every_n=2,
        conf_thr=0.3,
        max_subjects=1,
        logger=None,
        target_bbox=None,
        **kw,
    ):
        self.det = yolo_model
        self.pose = rtm_model
        self.nskip = max(1, int(process_every_n))
        self.conf_thr = float(conf_thr)
        self.tracker = BoxTracker(iou_thr=0.2, dist_thr=400)
        self.max_subjects = max_subjects
        self.logger = logger
        self.subject_states = {}
        self.cached_subjects = []
        self.last_known_box = None
        self.locked_id = None
        self.startup_frames = 0
        self.target_bbox = target_bbox
        self.target_initialized = False
        self.target_appearance = None
        self.frames_since_target_seen = 0
        self.max_frames_lost = 60

    def _get_or_create_state(self, tid):
        if tid not in self.subject_states:
            self.subject_states[tid] = {
                "contact": SkateContactAnalyzer(
                    FootIndices(L_ANKLE, L_HEEL, L_BIG_TOE),
                    FootIndices(R_ANKLE, R_HEEL, R_BIG_TOE),
                    min_conf=self.conf_thr,
                ),
                "lr_resolver": LRTemporalResolver(conf_thr=self.conf_thr),
                "L_knee": deque(maxlen=60),
                "R_knee": deque(maxlen=60),
            }
        return self.subject_states[tid]

    def _yolo_detect(self, frame):
        out = self.det(frame)
        boxes, scores = [], []
        if out is None:
            return boxes, scores

        raw_boxes = out.get("boxes") if isinstance(out, dict) else out
        if isinstance(raw_boxes, list):
            raw_boxes = np.array(raw_boxes)

        if raw_boxes is not None and len(raw_boxes) > 0:
            if raw_boxes.ndim == 1:
                raw_boxes = raw_boxes[None, :]
            for row in raw_boxes:
                boxes.append(row[:4].copy())
                scores.append(float(row[4]) if len(row) > 4 else 1.0)
        return boxes, scores

    def _rtm_infer(self, frame, boxes):
        if not boxes:
            return [], []
        bxs = [np.array(b).ravel()[:4].tolist() for b in boxes]
        try:
            res = self.pose(frame, bboxes=bxs)
        except Exception as e:
            logger.warning(f"RTM inference failed: {e}")
            return [], []

        k_list = (
            res[0]
            if isinstance(res, tuple)
            else (res.get("keypoints") if isinstance(res, dict) else res)
        )
        if k_list is None:
            logger.warning("RTM inference returned no keypoints")
            return [], []

        out_k, out_s = [], []

        for i in range(min(len(bxs), len(k_list))):
            k = np.array(k_list[i])
            if k.shape[1] >= 3:
                out_k.append(k[:, :2])
                out_s.append(k[:, 2])
            else:
                out_k.append(k[:, :2])
                out_s.append(np.ones(len(k)))
        return out_k, out_s

    def _get_global_state(self, L, R):
        l, r = (L if L != "Unknown" else "AIR"), (R if R != "Unknown" else "AIR")
        if l == "AIR" and r == "AIR":
            return "Airborne"
        if l != "AIR" and r == "AIR":
            return f"1-Foot Left ({l})"
        if l == "AIR" and r != "AIR":
            return f"1-Foot Right ({r})"
        return f"2-Foot ({l} - {r})"

    def process(self, frame: np.ndarray, fidx: int, ts: float) -> np.ndarray:
        H, W = frame.shape[:2]
        cx_img = W / 2.0

        do_proc = fidx % self.nskip == 0
        box_ids = []

        if do_proc:
            self.cached_subjects = []
            raw_boxes, raw_scores = self._yolo_detect(frame)

            if raw_boxes:
                clipped_boxes = [
                    np.clip(b, [0, 0, 0, 0], [W - 1, H - 1, W - 1, H - 1])
                    for b in raw_boxes
                ]

                min_height = H * 0.15
                valid_boxes = []
                for b in clipped_boxes:
                    h = b[3] - b[1]
                    if h > min_height:
                        valid_boxes.append(b)

                if not valid_boxes and clipped_boxes:
                    valid_boxes = clipped_boxes

                appearances = []
                for box in valid_boxes:
                    hist = extract_color_histogram(frame, box)
                    appearances.append(hist)

                box_ids = self.tracker.update(valid_boxes, frame.shape, appearances)
                candidates = list(zip(valid_boxes, box_ids))

                selected_candidate = None

                if not self.target_initialized and self.target_bbox is not None:
                    tx1, ty1, tx2, ty2 = self.target_bbox
                    txc = (tx1 + tx2) / 2
                    tyc = (ty1 + ty2) / 2
                    best_match = None
                    best_iou = 0.0

                    for box, tid in candidates:
                        iou = iou_xyxy([tx1, ty1, tx2, ty2], box)
                        if iou > best_iou:
                            best_iou = iou
                            best_match = (box, tid)

                    if best_match and best_iou > 0.3:
                        selected_candidate = best_match
                        self.locked_id = best_match[1]
                        self.target_initialized = True
                        self.frames_since_target_seen = 0
                        box, tid = best_match
                        self.last_known_box = box
                        idx = candidates.index(best_match)
                        if idx < len(appearances) and appearances[idx] is not None:
                            self.target_appearance = appearances[idx]
                            self.tracker.set_target(tid, appearances[idx], box)
                        logger.info(
                            f"Target initialized with ID {tid}, IoU={best_iou:.2f}"
                        )

                if selected_candidate is None and self.locked_id is not None:
                    for box, tid in candidates:
                        if tid == self.locked_id:
                            selected_candidate = (box, tid)
                            self.frames_since_target_seen = 0
                            break

                if selected_candidate is None and self.target_appearance is not None:
                    best_idx, best_sim = self.tracker.find_best_target_match(
                        valid_boxes, appearances, frame.shape
                    )
                    if best_idx is not None and best_idx >= 0 and best_sim > 0.5:
                        selected_candidate = (valid_boxes[best_idx], box_ids[best_idx])
                        self.locked_id = box_ids[best_idx]
                        self.frames_since_target_seen = 0
                        logger.info(
                            f"Reacquired target via appearance (sim={best_sim:.2f})"
                        )

                if selected_candidate is None and self.last_known_box is not None:
                    best_overlap_dist = float("inf")
                    for box, tid in candidates:
                        c_old = (
                            (self.last_known_box[0] + self.last_known_box[2]) / 2,
                            (self.last_known_box[1] + self.last_known_box[3]) / 2,
                        )
                        c_new = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                        dist = math.sqrt(
                            (c_old[0] - c_new[0]) ** 2 + (c_old[1] - c_new[1]) ** 2
                        )

                        if dist < 400:
                            if dist < best_overlap_dist:
                                best_overlap_dist = dist
                                selected_candidate = (box, tid)
                                self.locked_id = tid

                if selected_candidate is None and candidates:

                    def get_weighted_score(c):
                        box = c[0]
                        x1, y1, x2, y2 = box
                        area = (x2 - x1) * (y2 - y1)
                        bx = (x1 + x2) / 2.0
                        dist_norm = abs(bx - cx_img) / (W / 2.0)
                        centrality = 1.0 - (dist_norm**3)
                        centrality = max(0.001, centrality)
                        return area * centrality

                    candidates.sort(key=get_weighted_score, reverse=True)
                    selected_candidate = candidates[0]
                    self.locked_id = selected_candidate[1]
                    if not self.target_initialized:
                        self.target_initialized = True

                if selected_candidate:
                    box, tid = selected_candidate
                    self.last_known_box = box
                    if tid == self.locked_id:
                        self.frames_since_target_seen = 0
                    else:
                        self.frames_since_target_seen += 1

                    kps_list, scr_list = self._rtm_infer(frame, [box])
                    if kps_list:
                        self.cached_subjects.append(
                            {
                                "id": tid,
                                "bbox": box,
                                "kps": kps_list[0],
                                "scores": scr_list[0],
                            }
                        )
                        self.startup_frames += 1
                else:
                    self.frames_since_target_seen += 1

            active_tids = set(box_ids) if raw_boxes else set()
            for tid in list(self.subject_states.keys()):
                if tid not in active_tids:
                    del self.subject_states[tid]

        for subj_data in self.cached_subjects:
            tid, bbox, raw_kps, raw_scores = (
                subj_data["id"],
                subj_data["bbox"],
                subj_data["kps"],
                subj_data["scores"],
            )
            if raw_kps is None:
                continue

            state = self._get_or_create_state(tid)

            if do_proc:
                kps, scores = state["lr_resolver"].resolve(raw_kps, raw_scores)
                contact = state["contact"].update(kps, scores, ts)
                torso, _, _, _ = posture_metrics(kps, scores, self.conf_thr)
                lk = knee_angle(kps, scores, "left", self.conf_thr)
                rk = knee_angle(kps, scores, "right", self.conf_thr)
                if lk is not None:
                    state["L_knee"].append(lk)
                if rk is not None:
                    state["R_knee"].append(rk)
                state["cache"] = {
                    "kps": kps,
                    "scores": scores,
                    "contact": contact,
                    "torso": torso,
                    "lk": lk,
                    "rk": rk,
                }

            cache = state.get("cache", {})
            kps = cache.get("kps", raw_kps)
            scores = cache.get("scores", raw_scores)
            contact = cache.get(
                "contact",
                {
                    "left": {"state": "Unknown"},
                    "right": {"state": "Unknown"},
                    "wheeling": {"left": {}, "right": {}},
                },
            )
            state_str = self._get_global_state(
                contact["left"]["state"], contact["right"]["state"]
            )

            frame = self._draw_subject(
                frame,
                bbox,
                kps,
                scores,
                contact,
                cache.get("torso"),
                cache.get("lk"),
                cache.get("rk"),
                state_str,
                state,
            )

        return frame

    def _draw_subject(
        self, frame, bbox, kps, scores, out, torso, lk, rk, state_str, state
    ):
        x1, y1, x2, y2 = map(int, bbox[:4])
        # Draw BBox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw Skeleton
        draw_skeleton(frame, kps, scores, kpt_thr=self.conf_thr)

        # Posture Logic
        torso_val = int(torso) if torso is not None else 0
        posture = (
            "Upright"
            if torso_val <= 10
            else ("Slight Lean" if torso_val <= 30 else "Aggressive")
        )

        def wang(s):
            return f"{out['wheeling'][s].get('angle') or '-'}°"

        lines = [
            f"STATE: {state_str}",
            f"Posture: {posture} ({torso_val}°)",  # <--- ADDED ANGLE HERE
            f"L-Knee: {int(lk)}°" if lk is not None else "L-Knee: -",
            f"R-Knee: {int(rk)}°" if rk is not None else "R-Knee: -",
            f"L-Ankle: {wang('left')}",
            f"R-Ankle: {wang('right')}",
        ]

        # Fixed Dashboard Position (Top Left of Screen)
        start_x, start_y = 20, 40
        line_height = 25

        # Draw Background Panel
        overlay = frame.copy()
        bg_h = len(lines) * line_height + 40  # Added extra height for graphs
        cv2.rectangle(
            overlay,
            (start_x - 5, start_y - 25),
            (start_x + 250, start_y + bg_h),
            (0, 0, 0),
            -1,
        )
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Draw Text
        current_y = start_y
        for i, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (start_x, current_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            current_y += line_height

        # --- MOVED SPARKLINES TO DASHBOARD ---
        # Draw them below the text lines
        graph_y = current_y + 5

        # Left Knee Graph
        if state.get("L_knee"):
            self._sparkline(
                frame, list(state["L_knee"]), (start_x, graph_y, 100, 30), "Lk"
            )

        # Right Knee Graph (shifted right)
        if state.get("R_knee"):
            self._sparkline(
                frame, list(state["R_knee"]), (start_x + 110, graph_y, 100, 30), "Rk"
            )

        return frame

    @staticmethod
    def _sparkline(frame, vals, rect, label):
        x, y, w, h = rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 1)
        if not vals:
            return
        vs = np.array(vals)
        mn, mx = vs.min(), vs.max()
        if mx - mn < 1:
            mn, mx = mx - 1, mx + 1
        pts = []
        for i, v in enumerate(vs):
            px = x + 1 + int(i * (w - 2) / len(vs))
            py = y + h - 2 - int((v - mn) * (h - 4) / (mx - mn))
            pts.append((px, py))
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], (0, 255, 255), 1)
        cv2.putText(
            frame,
            label,
            (x + 2, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )


def detect_persons_first_frame(video_path, yolo_path, device="cpu"):
    """
    Detect persons in the first frame of a video.
    Returns list of bounding boxes with scores for user selection.
    """
    if YOLOX is None:
        raise ImportError("RTMLib not installed")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError("Failed to read first frame from video")

    H, W = frame.shape[:2]
    det = YOLOX(yolo_path, model_input_size=(640, 640), device=device)
    out = det(frame)

    boxes = []
    if out is not None:
        raw_boxes = out.get("boxes") if isinstance(out, dict) else out
        if isinstance(raw_boxes, list):
            raw_boxes = np.array(raw_boxes)

        if raw_boxes is not None and len(raw_boxes) > 0:
            if raw_boxes.ndim == 1:
                raw_boxes = raw_boxes[None, :]
            for row in raw_boxes:
                x1, y1, x2, y2 = row[:4]
                x1 = max(0, min(W, float(x1)))
                y1 = max(0, min(H, float(y1)))
                x2 = max(0, min(W, float(x2)))
                y2 = max(0, min(H, float(y2)))
                score = float(row[4]) if len(row) > 4 else 1.0
                box_h = y2 - y1
                if box_h > H * 0.10:
                    boxes.append(
                        {
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "score": round(score, 3),
                        }
                    )

    boxes.sort(
        key=lambda b: (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]),
        reverse=True,
    )
    return {"width": W, "height": H, "persons": boxes}


def analyze_video_v2(
    video_path,
    yolo_path,
    rtm_path,
    out_path,
    process_every_n=2,
    device="cpu",
    show_window=False,
    target_bbox=None,
):
    if YOLOX is None:
        raise ImportError("RTMLib not installed")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video FPS ({fps}), video may be corrupted")
    if W <= 0 or H <= 0:
        cap.release()
        raise RuntimeError(
            f"Invalid video dimensions ({W}x{H}), video may be corrupted"
        )

    det = YOLOX(yolo_path, model_input_size=(640, 640), device=device)
    pose = RTMPose(rtm_path, model_input_size=(288, 384), device=device)

    ana = SkatingVideoAnalyzer(
        det, pose, process_every_n=process_every_n, target_bbox=target_bbox
    )
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create video writer for: {out_path}")

    fidx = 0
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            out_frame = ana.process(frame, fidx, fidx / fps)
            writer.write(out_frame)
            fidx += 1
    finally:
        cap.release()
        writer.release()
