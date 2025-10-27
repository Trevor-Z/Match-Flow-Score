import json
import os
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ============================== CONFIG ===============================

SCRIPT_DIR = Path(__file__).resolve().parent

CSV_PATH = SCRIPT_DIR / "output" / "fight_frames.csv"
VIDEO_DIR = None  # None => use CSV paths as-is
SUMMARY_PATH = SCRIPT_DIR / "output" / "batch_summary.json"

FRAMES_ARE_ONE_BASED = False
SAMPLE_EVERY_N_FRAMES = 3

# ROIs
P1_COORD_MODE = "inner"           # "inner" | "left"
P1_FULL = (846, 69, 670, 1)       # P1 x is CENTER; bar extends LEFT
P2_FULL = (1074, 69, 670, 1)      # P2 x is LEFT (center side); bar extends RIGHT

P1_WIDTH = P1_FULL[2]
P2_WIDTH = P2_FULL[2]

# Debug output
SAVE_PER_VIDEO_CSV = True
OUT_CSV_DIR = SCRIPT_DIR / "output" / "px"
SAVE_DEBUG_STRIPS = False
DEBUG_STRIP_HEIGHT = 60
DEBUG_DIR = ''
DEBUG_SAVE_POLICY = "all"         # "accepted" | "all"

# Breadcrumbs / diagnostics
WRITE_BREADCRUMBS = False
WRITE_SCOPE_DIAGNOSTICS = False
PAINT_SCOPE_IN_DEBUG    = True
STRICT_DYNAMIC_REJECT   = False    # ignore 'bridged' for reject decision (non-yellow)

# HSV (OpenCV: H[0..179], S,V[0..255])
HSV_RED  = dict(H_MIN=160, H_MAX=179, S_MIN=105,  S_MAX=None, V_MIN=76, V_MAX=225)
HSV_BLUE = dict(H_MIN=95,  H_MAX=117, S_MIN=124,  S_MAX=None, V_MIN=88, V_MAX=None)
HSV_YELL_BRIGHT = dict(H_MIN=20, H_MAX=36, S_MIN=50,  S_MAX=None, V_MIN=215, V_MAX=None)

USE_YELLOW_INNER_GATE = True
YELLOW_ZONE_PX = 185

# Acceptance rule + scope
OUT_OF_RANGE_MAX_FRAC = 0.25
OUT_OF_RANGE_SCOPE = "dynamic"   # "full" | "inner" | "dynamic"
OUT_OF_RANGE_INNER_PX  = 220
OUT_OF_RANGE_MARGIN_PX = 8
OUT_OF_RANGE_MIN_SCOPE_PX = 15

LOW_HP_SCOPE_MODE = "disable_min"      # "off" | "override_min" | "disable_min"
OUT_OF_RANGE_MIN_SCOPE_PX_YELLOW = 0
OUT_OF_RANGE_MAX_FRAC_YELLOW = None

# Tiny-bar behavior
LOW_PERCENT_TURNOFF = 5.0
MIN_ACCEPT_LEADING_PX = 3

# Gap bridging (single-frame center-edge helper; also reused for "bridged" mask)
ENABLE_GAP_BRIDGE   = True
BRIDGE_GAP_MIN_PX   = 1
BRIDGE_GAP_MAX_PX   = 10
BRIDGE_TRAIL_MIN_PX = 10

# Cap dynamic margin for small bars
SMALL_BAR_MARGIN_THRESHOLD = 20
SMALL_BAR_MARGIN_MIN       = 2
SMALL_BAR_MARGIN_RATIO     = 0.5

# Leading-edge stabilization
LEADING_STICKY_RADIUS = 2
LEADING_HSV_TOL = dict(H=3, S=12, V=12)  # near-miss tolerance for index 0 only
ENABLE_FULL_BAR_GUARD = True             # ignore mask[0] in scope when leading >= width-1

# ===== Rejection rules =====
# 1) Post-hi gap followed by masked island
ENABLE_POST_HI_RULE   = True
POST_HI_GAP_MIN       = 6    # zero-run >= x after hi
POST_HI_ISLAND_MIN    = 12    # one-run >= y after that gap

# 2) Whole-bar internal gaps
ENABLE_INTERNAL_GAPS_RULE = True
MAX_INTERNAL_GAPS         = 3      # reject if internal gaps > this
INTERNAL_GAPS_COUNT_MODE  = "bridged"  # "raw" | "bridged"
INTERNAL_ISLAND_MIN_LEN   = 12      # NEW: both neighboring masked islands must be >= this length

# Post-round stabilization
ENABLE_POST_ROUND_FINALIZE    = True
POST_ROUND_SKIP_FRAMES        = 60
POST_ROUND_MAX_EXTRA_SAMPLES  = 20   # in addition to the initial two samples
POST_ROUND_ZERO_CONSECUTIVE   = 5    # consecutive zero-mask samples to zero out

# ============================== HELPERS ===============================

def _ensure_dir(d):
    if d and not os.path.exists(d): os.makedirs(d, exist_ok=True)

def _load_csv(path):
    for enc in ("utf-8-sig","utf-8","latin-1"):
        try: return pd.read_csv(path, encoding=enc)
        except Exception: pass
    return pd.read_csv(path)

def _resolve_video_path(name):
    name_path = Path(str(name))
    if VIDEO_DIR:
        base_dir = Path(VIDEO_DIR)
        if name_path.is_absolute():
            return str(name_path)
        return str(base_dir / name_path)
    return str(name_path)


def _load_batch_summary(summary_path):
    summary_path = Path(summary_path) if summary_path else None
    if not summary_path or not summary_path.exists():
        return {}
    try:
        with summary_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {}

    video_map = {}
    if isinstance(data, dict):
        data = data.get("summaries") or data.get("videos") or []
    if not isinstance(data, list):
        return {}

    for entry in data:
        if not isinstance(entry, dict):
            continue
        video_str = entry.get("video")
        if not video_str:
            continue
        video_key = Path(str(video_str)).name.lower()
        round_entries = entry.get("rounds_processed") or []
        durations = {}
        for item in round_entries:
            if not isinstance(item, dict):
                continue
            try:
                round_id = int(item.get("round"))
            except (TypeError, ValueError):
                continue
            if round_id <= 0:
                continue
            duration = item.get("duration_frames")
            if duration is None:
                continue
            try:
                duration_val = int(duration)
            except (TypeError, ValueError):
                continue
            if duration_val > 0:
                durations[round_id] = duration_val
        if durations:
            video_map[video_key] = durations
    return video_map


def _is_valid_number(value, allow_zero=True):
    try:
        val = float(value)
    except (TypeError, ValueError):
        return False
    if math.isnan(val):
        return False
    if not allow_zero and val <= 0:
        return False
    return True

def _progress_bar(done, total, prefix=""):
    total = max(1,total)
    w = 28
    filled = int(w * done / total)
    bar = "#"*filled + "-"*(w-filled)
    print(f"\r{prefix} [{bar}] {done}/{total}", end="")
    if done>=total: print()

def _make_debug_strip(line_bgr_oriented, mask_oriented, total_h=60, scope=None):
    if line_bgr_oriented is None or mask_oriented is None:
        return None
    W = line_bgr_oriented.shape[0]
    top_h = total_h // 2
    bot_h = total_h - top_h
    top = cv2.resize(line_bgr_oriented[np.newaxis, :, :], (W, top_h), interpolation=cv2.INTER_NEAREST)
    bot_gray = cv2.resize((mask_oriented * 255).astype(np.uint8)[np.newaxis, :], (W, bot_h), interpolation=cv2.INTER_NEAREST)
    bot = cv2.cvtColor(bot_gray, cv2.COLOR_GRAY2BGR)
    if PAINT_SCOPE_IN_DEBUG and scope is not None:
        lo, hi = scope
        lo = max(0, min(W - 1, int(lo)))
        hi = max(lo + 1, min(W, int(hi)))
        roi = bot[:, lo:hi, :]
        white = np.full_like(roi, 255, dtype=np.uint8)
        blended = cv2.addWeighted(roi, 0.3, white, 0.7, 0)
        bot[:, lo:hi, :] = blended
        bot[:, lo:lo+1, :] = (0, 0, 255)
        bot[:, hi-1:hi, :] = (0, 255, 0)
    return np.vstack([top, bot])

def _gate_color_mask(H, S, V, spec):
    base = (H >= spec["H_MIN"]) & (H <= spec["H_MAX"]) & (S >= spec["S_MIN"]) & (V >= spec["V_MIN"])
    smax_ok = True if ("S_MAX" not in spec or spec["S_MAX"] is None) else (S <= spec["S_MAX"])
    vmax_ok = True if ("V_MAX" not in spec or spec["V_MAX"] is None) else (V <= spec["V_MAX"])
    return base & smax_ok & vmax_ok

def _hsv_mask_union_for_side(bgr_line, side_key):
    hsv = cv2.cvtColor(bgr_line.reshape(1,-1,3), cv2.COLOR_BGR2HSV).reshape(-1,3)
    H,S,V = hsv[:,0], hsv[:,1], hsv[:,2]
    if side_key == "p1":
        healthy = _gate_color_mask(H,S,V, HSV_RED)
    else:
        healthy = _gate_color_mask(H,S,V, HSV_BLUE)
    yell = _gate_color_mask(H,S,V, HSV_YELL_BRIGHT)
    W = H.shape[0]
    if USE_YELLOW_INNER_GATE:
        inner = np.zeros(W, dtype=bool); inner[:min(W, YELLOW_ZONE_PX)] = True
        yellow_ok = yell & inner
    else:
        yellow_ok = yell
    mask = (healthy | yellow_ok).astype(np.uint8)
    return mask, hsv

def _extract_oriented_line_anchored(frame_bgr, base_roi, eff_w, direction, anchor):
    x,y,w,h = base_roi
    H,W = frame_bgr.shape[:2]
    use_w = int(max(1, min(int(eff_w), int(w))))
    x_eff = x + (w - use_w) if anchor == 'right' else x
    if x_eff < 0 or y < 0 or (x_eff + use_w) > W or (y + h) > H:
        return None
    crop = frame_bgr[y:y+h, x_eff:x_eff+use_w]
    if crop.size == 0: return None
    line = crop.mean(axis=0).astype(np.uint8)
    if direction == 'RL':
        line = line[::-1]
    return line

def _leading_simple(mask_oriented: np.ndarray) -> int:
    if mask_oriented is None or mask_oriented.size==0: return 0
    zeros = np.where(mask_oriented==0)[0]
    return int(mask_oriented.size if zeros.size==0 else zeros[0])

def _leading_with_bridge(mask_oriented: np.ndarray, gap_min: int, gap_max: int, trail_min: int):
    W = int(mask_oriented.size)
    if W == 0: return 0, False
    i = 0
    while i < W and mask_oriented[i] == 1: i += 1
    L = i
    if not ENABLE_GAP_BRIDGE: return L, False
    j = L; G = 0
    while j < W and mask_oriented[j] == 0 and G <= gap_max: G += 1; j += 1
    if G < gap_min or G > gap_max: return L, False
    R = 0; t = j
    while t < W and mask_oriented[t] == 1: R += 1; t += 1
    if R >= trail_min:
        Lp = min(j + R, W)
        return int(Lp), True
    return L, False

def _in_yellow_zone(leading_px: int, width_px: int) -> bool:
    return USE_YELLOW_INNER_GATE and (leading_px <= min(width_px, YELLOW_ZONE_PX))

# ---------- Leading-edge stabilization ----------

def _h_in_range(h, lo, hi):
    return (h >= lo) and (h <= hi)

def _near_miss_ok_hsv(hsv_px, side_key):
    H,S,V = int(hsv_px[0]), int(hsv_px[1]), int(hsv_px[2])
    tolH, tolS, tolV = LEADING_HSV_TOL["H"], LEADING_HSV_TOL["S"], LEADING_HSV_TOL["V"]
    if side_key == "p1":
        return _h_in_range(H, max(0, HSV_RED["H_MIN"] - tolH), min(HSV_RED["H_MAX"] + tolH, 179)) \
               and (S >= max(0, (HSV_RED["S_MIN"] - tolS))) and (V >= max(0, (HSV_RED["V_MIN"] - tolV))) \
               and (True if (HSV_RED.get("S_MAX") is None) else S <= (HSV_RED["S_MAX"] + tolS)) \
               and (True if (HSV_RED.get("V_MAX") is None) else V <= (HSV_RED["V_MAX"] + tolV))
    else:
        return _h_in_range(H, max(0, HSV_BLUE["H_MIN"] - tolH), min(HSV_BLUE["H_MAX"] + tolH, 179)) \
               and (S >= max(0, (HSV_BLUE["S_MIN"] - tolS))) and (V >= max(0, (HSV_BLUE["V_MIN"] - tolV))) \
               and (True if (HSV_BLUE.get("S_MAX") is None) else S <= (HSV_BLUE["S_MAX"] + tolS)) \
               and (True if (HSV_BLUE.get("V_MAX") is None) else V <= (HSV_BLUE["V_MAX"] + tolV))

def _stabilize_leading_edge(mask_oriented: np.ndarray, hsv_oriented: np.ndarray, side_key: str):
    if mask_oriented is None or mask_oriented.size == 0:
        return mask_oriented
    m = mask_oriented.copy()
    W = m.size
    if W == 0:
        return m
    R = min(LEADING_STICKY_RADIUS, max(0, W-1))
    if R > 0 and m[0] == 0:
        if np.any(m[1:R+1] == 1):
            m[0] = 1
    if m[0] == 0:
        hsv0 = hsv_oriented[0]
        if _near_miss_ok_hsv(hsv0, side_key):
            m[0] = 1
    return m

# -------------------- Runs / gap analysis -----------------------

def _runs01(mask: np.ndarray):
    """List of (value, start, length) contiguous runs across the full array."""
    res = []
    if mask is None or mask.size == 0: return res
    W = int(mask.size)
    cur_val = int(mask[0]); start = 0
    for i in range(1, W):
        v = int(mask[i])
        if v != cur_val:
            res.append((cur_val, start, i - start))
            cur_val, start = v, i
    res.append((cur_val, start, W - start))
    return res

def _mask_with_bridged_gaps(mask: np.ndarray, gap_min: int, gap_max: int, trail_min: int):
    """
    Build a 'bridged' version of the mask by filling zero-runs whose length
    is in [gap_min, gap_max] AND are strictly between one-runs.
    Optionally require the right one-run (after the gap) to be >= trail_min.
    """
    if mask is None or mask.size == 0: return mask
    runs = _runs01(mask)
    if not runs: return mask
    m = mask.copy()
    for i in range(1, len(runs)-1):
        v, st, ln = runs[i]
        if v != 0:  # only zero-runs
            continue
        if ln < gap_min or ln > gap_max:
            continue
        left_is_one  = (runs[i-1][0] == 1)
        right_is_one = (runs[i+1][0] == 1)
        if not (left_is_one and right_is_one):
            continue
        right_len = runs[i+1][2]
        if right_len < max(1, trail_min):
            continue
        m[st:st+ln] = 1
    return m

def _post_hi_gap_violation(mask: np.ndarray, hi: int, gap_min: int, island_min: int):
    """if there's a zero-run >= gap_min after hi, and then a one-run >= island_min after that gap => violation"""
    W = int(mask.size)
    hi = max(0, min(W, int(hi)))
    if hi >= W: return False, 0, 0
    runs = _runs01(mask)
    acc = 0; idx = 0
    while idx < len(runs) and acc + runs[idx][2] <= hi:
        acc += runs[idx][2]; idx += 1
    local_runs = []
    if idx < len(runs):
        val, st, ln = runs[idx]
        offset = hi - acc
        if offset > 0:
            tail_len = ln - offset
            if tail_len > 0:
                local_runs.append((val, hi, tail_len))
            idx += 1
        else:
            local_runs.append(runs[idx]); idx += 1
        for j in range(idx, len(runs)):
            local_runs.append(runs[j])
    else:
        return False, 0, 0

    gap_len_seen = 0
    for k, (v, st, ln) in enumerate(local_runs):
        if v == 0 and ln >= gap_min:
            gap_len_seen = ln
            for m in range(k+1, len(local_runs)):
                vv, ss, ll = local_runs[m]
                if vv == 1 and ll >= island_min:
                    return True, gap_len_seen, ll
            break
    return False, gap_len_seen, 0

def _count_internal_gaps_from_mask(mask: np.ndarray, island_min_len: int):
    """
    Count zero-runs strictly between one-runs (ignore leading/trailing zeros),
    BUT only count a zero-run if BOTH neighboring one-runs have length >= island_min_len.
    """
    runs = _runs01(mask)
    if not runs: return 0
    one_idxs = [i for i,(v,_,_) in enumerate(runs) if v == 1]
    if not one_idxs: return 0
    first_one, last_one = one_idxs[0], one_idxs[-1]
    internal = 0
    for i in range(first_one, last_one):
        v, st, ln = runs[i]
        if v == 0:
            # neighbors exist because i is between first_one and last_one
            left_len  = runs[i-1][2]
            right_len = runs[i+1][2]
            if left_len >= island_min_len and right_len >= island_min_len:
                internal += 1
    return internal

def _count_internal_gaps(mask: np.ndarray, mode: str, island_min_len: int):
    """
    mode: "raw"     -> count on original mask
          "bridged" -> first fill bridgeable gaps, then count
    island_min_len: enforce across *all* islands by pruning short ones first
    """
    # 1) choose base mask
    if mode.lower() == "bridged":
        m = _mask_with_bridged_gaps(mask, BRIDGE_GAP_MIN_PX, BRIDGE_GAP_MAX_PX, BRIDGE_TRAIL_MIN_PX)
    else:
        m = mask

    # 2) prune all islands shorter than the threshold
    m = _prune_short_islands(m, island_min_len)

    # 3) count internal gaps between remaining islands
    #    (now every island is >= island_min_len, so no extra neighbor checks needed)
    runs = _runs01(m)
    if not runs:
        return 0
    one_idxs = [i for i,(v,_,_) in enumerate(runs) if v == 1]
    if not one_idxs:
        return 0
    first_one, last_one = one_idxs[0], one_idxs[-1]
    internal = 0
    for i in range(first_one, last_one):
        v, st, ln = runs[i]
        if v == 0:
            internal += 1
    return internal

def _prune_short_islands(mask: np.ndarray, min_len: int) -> np.ndarray:
    """
    Turn every 1-run shorter than min_len into zeros, so *all* remaining islands
    meet the minimum. This is applied before counting internal gaps.
    """
    if mask is None or mask.size == 0 or min_len <= 1:
        return mask
    m = mask.copy()
    for v, st, ln in _runs01(m):
        if v == 1 and ln < min_len:
            m[st:st+ln] = 0
    return m

# ---------------------------------------------------------------

def _scope_fraction(mask: np.ndarray, leading: int, width_px: int):
    W = width_px
    in_yel = _in_yellow_zone(leading, W)
    if W <= 0: return 0.0, in_yel, 0, 0

    pct = (100.0 * leading / float(W))

    if (pct <= LOW_PERCENT_TURNOFF) or (leading < MIN_ACCEPT_LEADING_PX):
        lo, hi = 0, min(W, max(1, int(leading)))
    else:
        if OUT_OF_RANGE_SCOPE == "full":
            lo, hi = 0, W
        elif OUT_OF_RANGE_SCOPE == "inner":
            lo, hi = 0, min(W, OUT_OF_RANGE_INNER_PX)
        else:  # dynamic
            floor_px = OUT_OF_RANGE_MIN_SCOPE_PX
            if in_yel and LOW_HP_SCOPE_MODE != "off":
                if LOW_HP_SCOPE_MODE == "override_min":
                    floor_px = OUT_OF_RANGE_MIN_SCOPE_PX_YELLOW
                elif LOW_HP_SCOPE_MODE == "disable_min":
                    floor_px = 0
            eff_margin = OUT_OF_RANGE_MARGIN_PX
            if leading < SMALL_BAR_MARGIN_THRESHOLD:
                cap_by_ratio = int(max(1, leading * float(SMALL_BAR_MARGIN_RATIO)))
                eff_margin = min(OUT_OF_RANGE_MARGIN_PX, max(SMALL_BAR_MARGIN_MIN, cap_by_ratio))
            win = max(int(leading) + int(eff_margin), int(floor_px))
            lo, hi = 0, min(W, win)

    sub = mask[lo:hi]
    if ENABLE_FULL_BAR_GUARD and (leading >= (width_px - 1)) and (sub.size >= 2):
        sub = sub[1:]
    frac = float(sub.sum()) / float(max(1, sub.size))
    return frac, in_yel, lo, hi

# --- Globals set in main after canonicalization ---
P1_ROI_USED = None
P2_ROI_USED = None

def _measure_both_from_capture(cap, frame_idx):
    """
    Convenience wrapper that positions the capture at frame_idx and measures both sides.
    Returns the same tuple as _measure_both_from_frame, or None on failure.
    """
    if frame_idx < 0 or cap is None:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = cap.read()
    if not ret:
        return None
    return _measure_both_from_frame(frame)

def _determine_post_round_state(cap, end_frame_idx, step, skip_frames, max_extra_samples):
    """
    After a round ends, probe additional samples to detect sustained zero-mask regions.
    Returns a dict marking sides that should be zeroed when the configured condition holds.
    """
    if not ENABLE_POST_ROUND_FINALIZE or cap is None:
        return {}
    step = max(1, int(step))
    skip_frames = max(0, int(skip_frames))
    max_extra_samples = max(0, int(max_extra_samples))
    zero_needed = max(1, int(POST_ROUND_ZERO_CONSECUTIVE))
    start_idx = max(0, int(end_frame_idx) + skip_frames)
    max_samples = max(zero_needed, 2 + max_extra_samples)

    zero_counts = {"p1": 0, "p2": 0}
    zero_flags = {"p1": False, "p2": False}

    for sample_idx in range(max_samples):
        frame_idx = start_idx + sample_idx * step
        measured = _measure_both_from_capture(cap, frame_idx)
        if measured is None:
            break
        _, _, _, _, meta1, meta2 = measured

        def _update_zero(meta, key):
            if meta.get('roi_oob', False):
                zero_counts[key] = 0
                return
            leading = int(meta.get('leading', 0))
            if leading <= 0:
                zero_counts[key] += 1
                if zero_counts[key] >= zero_needed:
                    zero_flags[key] = True
            else:
                zero_counts[key] = 0

        _update_zero(meta1, "p1")
        _update_zero(meta2, "p2")

        if zero_flags["p1"] and zero_flags["p2"]:
            break

    return {k: v for k, v in zero_flags.items() if v}

def _measure_both_from_frame(frame):
    """
    Returns: (p1px, p2px, viz1, viz2, p1_meta, p2_meta)
    """
    # ---- P1 ----
    line1 = _extract_oriented_line_anchored(frame, P1_ROI_USED, P1_WIDTH, direction='RL', anchor='right')
    if line1 is None:
        p1px, viz1, p1_meta = 0, None, {'roi_oob': True}
    else:
        m1, hsv1 = _hsv_mask_union_for_side(line1, "p1")
        m1 = _stabilize_leading_edge(m1, hsv1, side_key="p1")
        lead1, bridged1 = (_leading_with_bridge(m1, BRIDGE_GAP_MIN_PX, BRIDGE_GAP_MAX_PX, BRIDGE_TRAIL_MIN_PX)
                           if ENABLE_GAP_BRIDGE else (_leading_simple(m1), False))
        if ((100.0*lead1/P1_WIDTH) <= LOW_PERCENT_TURNOFF) or (lead1 < MIN_ACCEPT_LEADING_PX):
            frac1, y1, lo1, hi1 = (1.0 if lead1>0 else 0.0, _in_yellow_zone(lead1, P1_WIDTH), 0, max(1, lead1))
        else:
            frac1, y1, lo1, hi1 = _scope_fraction(m1, lead1, P1_WIDTH)

        max_oof_1 = OUT_OF_RANGE_MAX_FRAC_YELLOW if (y1 and OUT_OF_RANGE_MAX_FRAC_YELLOW is not None) else OUT_OF_RANGE_MAX_FRAC
        oof1 = 1.0 - float(frac1)
        if STRICT_DYNAMIC_REJECT and not y1:
            bridged1 = False
        pass1 = bridged1 or (oof1 <= max_oof_1)

        # New rules
        post_hi_violation_1, post_gap_len_1, post_island_len_1 = (False, 0, 0)
        if ENABLE_POST_HI_RULE:
            post_hi_violation_1, post_gap_len_1, post_island_len_1 = _post_hi_gap_violation(m1, hi1, POST_HI_GAP_MIN, POST_HI_ISLAND_MIN)

        internal_gaps_raw_1 = _count_internal_gaps(m1, mode="raw",     island_min_len=INTERNAL_ISLAND_MIN_LEN) if ENABLE_INTERNAL_GAPS_RULE else 0
        internal_gaps_br_1  = _count_internal_gaps(m1, mode="bridged", island_min_len=INTERNAL_ISLAND_MIN_LEN) if ENABLE_INTERNAL_GAPS_RULE else 0
        enforce_gaps_1 = internal_gaps_br_1 if INTERNAL_GAPS_COUNT_MODE == "bridged" else internal_gaps_raw_1

        if (ENABLE_POST_HI_RULE and post_hi_violation_1) or (ENABLE_INTERNAL_GAPS_RULE and (MAX_INTERNAL_GAPS is not None) and (enforce_gaps_1 > MAX_INTERNAL_GAPS)):
            pass1 = False

        p1px = int(lead1) if pass1 else 0
        viz1 = _make_debug_strip(line1, m1, DEBUG_STRIP_HEIGHT, scope=(lo1, hi1)) if SAVE_DEBUG_STRIPS else None
        p1_meta = {
            'roi_oob': False, 'bridged': bool(bridged1), 'oof': float(oof1),
            'max_oof': float(max_oof_1), 'passed': bool(pass1),
            'leading': int(lead1), 'frac': float(frac1), 'lo': int(lo1), 'hi': int(hi1),
            'post_hi_violation': bool(post_hi_violation_1),
            'post_hi_gap_len': int(post_gap_len_1),
            'post_hi_island_len': int(post_island_len_1),
            'internal_gaps_raw': int(internal_gaps_raw_1),
            'internal_gaps_bridged': int(internal_gaps_br_1),
            'internal_gaps_mode': INTERNAL_GAPS_COUNT_MODE,
            'internal_island_min_len': int(INTERNAL_ISLAND_MIN_LEN),
        }

    # ---- P2 ----
    line2 = _extract_oriented_line_anchored(frame, P2_ROI_USED, P2_WIDTH, direction='LR', anchor='left')
    if line2 is None:
        p2px, viz2, p2_meta = 0, None, {'roi_oob': True}
    else:
        m2, hsv2 = _hsv_mask_union_for_side(line2, "p2")
        m2 = _stabilize_leading_edge(m2, hsv2, side_key="p2")
        lead2, bridged2 = (_leading_with_bridge(m2, BRIDGE_GAP_MIN_PX, BRIDGE_GAP_MAX_PX, BRIDGE_TRAIL_MIN_PX)
                           if ENABLE_GAP_BRIDGE else (_leading_simple(m2), False))
        if ((100.0*lead2/P2_WIDTH) <= LOW_PERCENT_TURNOFF) or (lead2 < MIN_ACCEPT_LEADING_PX):
            frac2, y2, lo2, hi2 = (1.0 if lead2>0 else 0.0, _in_yellow_zone(lead2, P2_WIDTH), 0, max(1, lead2))
        else:
            frac2, y2, lo2, hi2 = _scope_fraction(m2, lead2, P2_WIDTH)

        max_oof_2 = OUT_OF_RANGE_MAX_FRAC_YELLOW if (y2 and OUT_OF_RANGE_MAX_FRAC_YELLOW is not None) else OUT_OF_RANGE_MAX_FRAC
        oof2 = 1.0 - float(frac2)
        if STRICT_DYNAMIC_REJECT and not y2:
            bridged2 = False
        pass2 = bridged2 or (oof2 <= max_oof_2)

        post_hi_violation_2, post_gap_len_2, post_island_len_2 = (False, 0, 0)
        if ENABLE_POST_HI_RULE:
            post_hi_violation_2, post_gap_len_2, post_island_len_2 = _post_hi_gap_violation(m2, hi2, POST_HI_GAP_MIN, POST_HI_ISLAND_MIN)

        internal_gaps_raw_2 = _count_internal_gaps(m2, mode="raw",     island_min_len=INTERNAL_ISLAND_MIN_LEN) if ENABLE_INTERNAL_GAPS_RULE else 0
        internal_gaps_br_2  = _count_internal_gaps(m2, mode="bridged", island_min_len=INTERNAL_ISLAND_MIN_LEN) if ENABLE_INTERNAL_GAPS_RULE else 0
        enforce_gaps_2 = internal_gaps_br_2 if INTERNAL_GAPS_COUNT_MODE == "bridged" else internal_gaps_raw_2

        if (ENABLE_POST_HI_RULE and post_hi_violation_2) or (ENABLE_INTERNAL_GAPS_RULE and (MAX_INTERNAL_GAPS is not None) and (enforce_gaps_2 > MAX_INTERNAL_GAPS)):
            pass2 = False

        p2px = int(lead2) if pass2 else 0
        viz2 = _make_debug_strip(line2, m2, DEBUG_STRIP_HEIGHT, scope=(lo2, hi2)) if SAVE_DEBUG_STRIPS else None
        p2_meta = {
            'roi_oob': False, 'bridged': bool(bridged2), 'oof': float(oof2),
            'max_oof': float(max_oof_2), 'passed': bool(pass2),
            'leading': int(lead2), 'frac': float(frac2), 'lo': int(lo2), 'hi': int(hi2),
            'post_hi_violation': bool(post_hi_violation_2),
            'post_hi_gap_len': int(post_gap_len_2),
            'post_hi_island_len': int(post_island_len_2),
            'internal_gaps_raw': int(internal_gaps_raw_2),
            'internal_gaps_bridged': int(internal_gaps_br_2),
            'internal_gaps_mode': INTERNAL_GAPS_COUNT_MODE,
            'internal_island_min_len': int(INTERNAL_ISLAND_MIN_LEN),
        }

    return p1px, p2px, viz1, viz2, p1_meta, p2_meta

def _parse_round_specs(row, summary_durations=None):
    rounds = []
    if hasattr(row, "index"):
        columns_lower = {str(col).lower(): col for col in row.index}
    else:
        columns_lower = {}
    summary_durations = summary_durations or {}

    for rid in (1, 2, 3):
        frame_col_name = columns_lower.get(f"frame_r{rid}")
        if frame_col_name is None:
            frame_col_name = columns_lower.get(f"frame r{rid}")
        if frame_col_name is None:
            continue

        frame_val = row[frame_col_name]
        if not _is_valid_number(frame_val):
            continue

        duration = None
        duration_col_name = columns_lower.get(f"r{rid}_duration") or columns_lower.get(f"r{rid} duration")
        if duration_col_name is not None:
            duration_val = row[duration_col_name]
            if _is_valid_number(duration_val, allow_zero=False):
                duration = int(float(duration_val))

        if duration is None:
            duration = summary_durations.get(rid)

        if duration is None:
            continue

        rounds.append((rid, int(float(frame_val)), int(duration)))
    return rounds

# ============================== CORE ===============================

def process_round_fast(cap, video_basename, round_id, start_frame, duration, step, one_based, dbg_root):
    rows = []

    s = int(start_frame) - (1 if one_based else 0)
    if s < 0: s = 0
    end_f = s + int(duration)

    sample_idxs = list(range(s, end_f, step))
    ns = len(sample_idxs)
    if ns == 0: return rows

    _ensure_dir(dbg_root)
    if SAVE_DEBUG_STRIPS:
        _ensure_dir(os.path.join(dbg_root, video_basename, f"round{round_id}"))

    cap.set(cv2.CAP_PROP_POS_FRAMES, s)
    next_i = 0; curr = s; done = 0

    while curr < end_f and next_i < ns:
        ret, frame = cap.read()
        if not ret: break
        target = sample_idxs[next_i]
        if curr == target:
            p1px, p2px, v1, v2, meta1, meta2 = _measure_both_from_frame(frame)

            p1_status, p1_reason = ("accepted", "") if (not meta1.get('roi_oob', False) and meta1.get('passed', False)) \
                                   else ("rejected", ("roi_oob" if meta1.get('roi_oob', False)
                                                      else f"scope_oof:1-frac={meta1.get('oof',0):.2f}>{meta1.get('max_oof',OUT_OF_RANGE_MAX_FRAC):.2f}"))
            p2_status, p2_reason = ("accepted", "") if (not meta2.get('roi_oob', False) and meta2.get('passed', False)) \
                                   else ("rejected", ("roi_oob" if meta2.get('roi_oob', False)
                                                      else f"scope_oof:1-frac={meta2.get('oof',0):.2f}>{meta2.get('max_oof',OUT_OF_RANGE_MAX_FRAC):.2f}"))

            # annotate new-rule reasons if applicable
            if p1_status == "rejected" and not meta1.get('roi_oob', False):
                if meta1.get('post_hi_violation', False):
                    p1_reason = f"post_hi_gap>= {POST_HI_GAP_MIN} & island>= {POST_HI_ISLAND_MIN}"
                else:
                    if ENABLE_INTERNAL_GAPS_RULE and (MAX_INTERNAL_GAPS is not None):
                        raw = meta1.get('internal_gaps_raw',0); br = meta1.get('internal_gaps_bridged',0)
                        used = br if INTERNAL_GAPS_COUNT_MODE=="bridged" else raw
                        if used > MAX_INTERNAL_GAPS:
                            p1_reason = f"internal_gaps({INTERNAL_GAPS_COUNT_MODE}, island_min={INTERNAL_ISLAND_MIN_LEN})={used}>{MAX_INTERNAL_GAPS} [raw={raw}, bridged={br}]"
            if p2_status == "rejected" and not meta2.get('roi_oob', False):
                if meta2.get('post_hi_violation', False):
                    p2_reason = f"post_hi_gap>= {POST_HI_GAP_MIN} & island>= {POST_HI_ISLAND_MIN}"
                else:
                    if ENABLE_INTERNAL_GAPS_RULE and (MAX_INTERNAL_GAPS is not None):
                        raw = meta2.get('internal_gaps_raw',0); br = meta2.get('internal_gaps_bridged',0)
                        used = br if INTERNAL_GAPS_COUNT_MODE=="bridged" else raw
                        if used > MAX_INTERNAL_GAPS:
                            p2_reason = f"internal_gaps({INTERNAL_GAPS_COUNT_MODE}, island_min={INTERNAL_ISLAND_MIN_LEN})={used}>{MAX_INTERNAL_GAPS} [raw={raw}, bridged={br}]"

            out_idx = sample_idxs[next_i] + (1 if FRAMES_ARE_ONE_BASED else 0)

            if SAVE_DEBUG_STRIPS and DEBUG_SAVE_POLICY in ("all","accepted"):
                dbg_dir = os.path.join(dbg_root, video_basename, f"round{round_id}")
                _ensure_dir(dbg_dir)
                if v1 is not None and (DEBUG_SAVE_POLICY=="all" or p1_status=="accepted"):
                    lab = f"{p1px:04d}" if p1_status=="accepted" else "rej"
                    cv2.imwrite(os.path.join(dbg_dir, f"frame{out_idx}__p1__px{lab}__{p1_status}.png"), v1)
                if v2 is not None and (DEBUG_SAVE_POLICY=="all" or p2_status=="accepted"):
                    lab = f"{p2px:04d}" if p2_status=="accepted" else "rej"
                    cv2.imwrite(os.path.join(dbg_dir, f"frame{out_idx}__p2__px{lab}__{p2_status}.png"), v2)

            row = dict(
                video=video_basename, round=round_id, frame=out_idx,
                p1_status=p1_status, p1_px=int(p1px),
                p2_status=p2_status, p2_px=int(p2px),
            )
            if WRITE_BREADCRUMBS:
                row["p1_reason"] = p1_reason; row["p2_reason"] = p2_reason
                if WRITE_SCOPE_DIAGNOSTICS:
                    row.update(dict(
                        p1_leading=meta1.get('leading',0), p1_frac=meta1.get('frac',0.0), p1_lo=meta1.get('lo',0), p1_hi=meta1.get('hi',0),
                        p1_bridged=meta1.get('bridged',False),
                        p1_post_hi_gap_len=meta1.get('post_hi_gap_len',0), p1_post_hi_island_len=meta1.get('post_hi_island_len',0),
                        p1_post_hi_violation=meta1.get('post_hi_violation',False),
                        p1_internal_gaps_raw=meta1.get('internal_gaps_raw',0),
                        p1_internal_gaps_bridged=meta1.get('internal_gaps_bridged',0),
                        p1_internal_gaps_mode=meta1.get('internal_gaps_mode',"raw"),
                        p1_internal_island_min_len=meta1.get('internal_island_min_len',1),

                        p2_leading=meta2.get('leading',0), p2_frac=meta2.get('frac',0.0), p2_lo=meta2.get('lo',0), p2_hi=meta2.get('hi',0),
                        p2_bridged=meta2.get('bridged',False),
                        p2_post_hi_gap_len=meta2.get('post_hi_gap_len',0), p2_post_hi_island_len=meta2.get('post_hi_island_len',0),
                        p2_post_hi_violation=meta2.get('post_hi_violation',False),
                        p2_internal_gaps_raw=meta2.get('internal_gaps_raw',0),
                        p2_internal_gaps_bridged=meta2.get('internal_gaps_bridged',0),
                        p2_internal_gaps_mode=meta2.get('internal_gaps_mode',"raw"),
                        p2_internal_island_min_len=meta2.get('internal_island_min_len',1),
                    ))
            rows.append(row)

            done += 1; _progress_bar(done, ns, prefix=f"R{round_id} {video_basename}")
            next_i += 1

        curr += 1

    if rows and ENABLE_POST_ROUND_FINALIZE:
        zero_flags = _determine_post_round_state(cap, end_f, step, POST_ROUND_SKIP_FRAMES, POST_ROUND_MAX_EXTRA_SAMPLES)
        if zero_flags:
            last_row = rows[-1]
            if zero_flags.get("p1"):
                last_row['p1_status'] = "accepted"
                last_row['p1_px'] = 0
                if WRITE_BREADCRUMBS and 'p1_reason' in last_row:
                    last_row['p1_reason'] = ""
            if zero_flags.get("p2"):
                last_row['p2_status'] = "accepted"
                last_row['p2_px'] = 0
                if WRITE_BREADCRUMBS and 'p2_reason' in last_row:
                    last_row['p2_reason'] = ""

    return rows

def _save_per_video_round(csv_dir, video_path, round_id, rows):
    if not rows:
        print(f"\n[WARN] No rows kept for {os.path.basename(video_path)} R{round_id}.")
        return
    cols = ["video","round","frame","p1_status","p1_px","p2_status","p2_px"]
    if WRITE_BREADCRUMBS:
        cols += ["p1_reason","p2_reason"]
        if WRITE_SCOPE_DIAGNOSTICS:
            cols += [
                "p1_leading","p1_frac","p1_lo","p1_hi","p1_bridged",
                "p1_post_hi_gap_len","p1_post_hi_island_len","p1_post_hi_violation",
                "p1_internal_gaps_raw","p1_internal_gaps_bridged","p1_internal_gaps_mode","p1_internal_island_min_len",
                "p2_leading","p2_frac","p2_lo","p2_hi","p2_bridged",
                "p2_post_hi_gap_len","p2_post_hi_island_len","p2_post_hi_violation",
                "p2_internal_gaps_raw","p2_internal_gaps_bridged","p2_internal_gaps_mode","p2_internal_island_min_len",
            ]
    df = pd.DataFrame(rows, columns=cols)
    _ensure_dir(csv_dir)
    base_video = os.path.basename(video_path)
    out_path = os.path.join(csv_dir, f"{base_video}__round{round_id}_health_px.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] Wrote {out_path}")

# ============================== MAIN ===============================

def _canon_roi_for_p1(roi, coord_mode):
    x,y,w,h = roi
    if coord_mode.lower() == "inner":
        return (x - w, y, w, h)
    return roi

def main():
    global P1_WIDTH, P2_WIDTH, P1_ROI_USED, P2_ROI_USED
    P1_ROI_USED = _canon_roi_for_p1(P1_FULL, P1_COORD_MODE)
    P2_ROI_USED = P2_FULL
    P1_WIDTH = max(1, min(int(P1_WIDTH), int(P1_ROI_USED[2])))
    P2_WIDTH = max(1, min(int(P2_WIDTH), int(P2_ROI_USED[2])))

    df = _load_csv(CSV_PATH)
    if df is None or df.empty:
        print("[ERR] CSV read failed or is empty."); return

    duration_lookup = _load_batch_summary(SUMMARY_PATH)

    for r, row in df.iterrows():
        vpath = _resolve_video_path(row.iloc[0])
        base = os.path.basename(vpath)
        if not os.path.exists(vpath):
            print(f"[ERROR][Row {r}] Video not found: {vpath}")
            continue
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"[ERROR][Row {r}] Could not open video: {vpath}")
            continue

        video_key = Path(base).name.lower()
        summary_durations = duration_lookup.get(video_key, {})
        if not summary_durations and hasattr(row, "iloc"):
            original_key = Path(str(row.iloc[0])).name.lower()
            summary_durations = duration_lookup.get(original_key, {})

        round_specs = _parse_round_specs(row, summary_durations)
        if not round_specs:
            print(f"[WARN][Row {r}] No round specs for {base}")
            cap.release(); continue

        pretty = ", ".join([f"R{rid}@{sf}+{dur}" for (rid,sf,dur) in round_specs])
        print(f"[INFO][Row {r}] {base} -> rounds: {pretty}")

        for (rid, start_f, duration) in round_specs:
            rows = process_round_fast(cap, base, rid, start_f, duration,
                                      SAMPLE_EVERY_N_FRAMES, FRAMES_ARE_ONE_BASED, DEBUG_DIR)
            if SAVE_PER_VIDEO_CSV: _save_per_video_round(OUT_CSV_DIR, vpath, rid, rows)

        cap.release()

if __name__ == "__main__":
    main()
