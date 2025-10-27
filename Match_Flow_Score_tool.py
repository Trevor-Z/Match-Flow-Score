#!/usr/bin/env python3
"""
Match_Flow_Calculator.py

Pipeline:
1. Detect round start frames ("Fight") using the fight-frame classifier.
2. Detect round end frames using the round-end digit probe logic.
3. Extract health bar pixel values frame-by-frame for each round.
4. Convert pixel readings into percentages following the pct post-processing rules.
5. Write simple per-round CSVs (`frame`, `p1_pct`, `p2_pct`) plus optional metadata.
6. Aggregate the per-round CSVs into a match-flow score report.

The script can process a single video file or every video in a folder. Configure
paths directly in the SETTINGS block below; no command-line arguments are used.
This orchestrator reuses the project modules without modifying them.
"""

from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import pandas as pd
import torch
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# ---------------------------- USER SETTINGS ----------------------------- #
# Update the values below to point at your videos, model, and preferences.

SETTINGS = {
    # Required input (local override; no direct counterpart in helper scripts)
    "input_path": Path(r""),

    # Output folder for per-video artifacts and reports
    "output_dir": Path("output"),

    # Optional folder to dump debug captures for detected round ends (None disables)
    "round_end_debug_dir": None,

    # Match-finder model; defaults to MODEL_PATH in 4 Fight_Frame_Finder_v1.1.py
    "model_path": Path("model.pth"),

    # Max rounds per video (mirrors parse_rounds_from_filename logic in fight finder)
    "max_rounds": 3,

    # Round-end detection side selection (auto mirrors EndingFinder default)
    "side": "auto",
    
    "round1_start": 950,
    "round2_start": 2400,
    "round3_offset_after_r2": 2300, # When r2 was found, search r3 from (r2 + this)
    "round3_early_fallback": 500, # Try 500 frames earlier if round 3 not found

    # Frames to scan after fight frame; derived from capture tail handling in 6_sf6_input_extractor.py
    "max_round_frames": 7200,

    # Search offset between rounds (aligns with ROUND3_OFFSET logic in fight finder)
    "min_gap": 480,

    # Health sampling stride (override for SAMPLE_EVERY_N_FRAMES in 12_SF6_health_bar_extractor_px.py)
    "sample_step": 3,

    # Torch device override; None defers to auto detection
    "device": None,

    # Video extensions to consider when scanning folders
    "extensions": (".mp4", ".mkv", ".avi", ".mov"),

    # Folder recursion toggle for batch processing
    "recursive": False,

    # Tail length for round-end digit probe (defaults to DEFAULT_TAIL_CONSECUTIVE_NONE in EndingFinder.py)
    "round_end_tail": 10,
}

REPORT_FILE_GLOB = "*.csv"

REPORT_WEIGHTS = {
    "w_duration": 0.35,
    "w_health": 0.35,
    "w_streak": 0.2,
    "w_tempo": 0.1,
}

REPORT_WEIGHT_RANGES = {
    "w_duration": (0.0, 1.0),
    "w_health": (0.0, 1.0),
    "w_streak": (0.0, 1.0),
    "w_tempo": (0.0, 1.0),
}

REPORT_ENFORCE_SUM_TO_ONE = False
REPORT_T_MAX = 6000
REPORT_USE_PCT_WINNER = True


# --------------------------------------------------------------------------- #
# Module loading utilities
# --------------------------------------------------------------------------- #


def _load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required module '{path}' was not found.")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@dataclass
class ProjectModules:
    """Holds handles to the helper modules we depend on."""

    fight: object
    round_end: object
    health_px: object
    health_pct: object

    @staticmethod
    def load(base_dir: Path) -> "ProjectModules":
        fight = _load_module(
            "sf6_fight_finder",
            base_dir / "1_Fight_Frame_Finder.py",
        )
        round_end = _load_module(
            "sf6_round_end_finder",
            base_dir / "2_EndingFinder.py",
        )
        health_px = _load_module(
            "sf6_health_px",
            base_dir / "3_Health_Bar_Extractor_px.py",
        )
        health_pct = _load_module(
            "sf6_health_pct",
            base_dir / "4_Health_Bar_Extractor_pct.py",
        )
        return ProjectModules(
            fight=fight,
            round_end=round_end,
            health_px=health_px,
            health_pct=health_pct,
        )


# --------------------------------------------------------------------------- #
# Fight frame detector setup
# --------------------------------------------------------------------------- #


def _create_fight_model(mod, model_path: Path, device: torch.device):
    checkpoint = torch.load(str(model_path), map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model = mod.CustomSqueezeNet(num_classes=2).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    if device.type == "cuda":
        try:
            model = torch.jit.script(model)
        except Exception:
            pass

    transform = mod.transforms.Compose(
        [
            mod.transforms.Resize(
                (mod.RESIZE_HEIGHT, mod.RESIZE_WIDTH),
                interpolation=mod.transforms.InterpolationMode.BILINEAR,
            ),
            mod.transforms.ToTensor(),
            mod.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return model, transform


# Health extraction helpers
# --------------------------------------------------------------------------- #


def configure_health_px_module(mod):
    """Mimic the setup done in the original px extractor."""
    mod.P1_ROI_USED = mod._canon_roi_for_p1(mod.P1_FULL, mod.P1_COORD_MODE)
    mod.P2_ROI_USED = mod.P2_FULL
    mod.P1_WIDTH = max(1, min(int(mod.P1_WIDTH), int(mod.P1_ROI_USED[2])))
    mod.P2_WIDTH = max(1, min(int(mod.P2_WIDTH), int(mod.P2_ROI_USED[2])))


def extract_health_px_rows(
    mod,
    video_path: Path,
    video_basename: str,
    round_id: int,
    start_frame: int,
    end_frame: int,
    sample_step: int,
) -> List[Dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        duration = max(1, int(end_frame - start_frame + 1))
        rows = mod.process_round_fast(
            cap=cap,
            video_basename=video_basename,
            round_id=round_id,
            start_frame=start_frame,
            duration=duration,
            step=max(1, int(sample_step)),
            one_based=mod.FRAMES_ARE_ONE_BASED,
            dbg_root=mod.DEBUG_DIR if getattr(mod, "SAVE_DEBUG_STRIPS", False) else None,
        )
        return rows
    finally:
        cap.release()


def build_pct_outputs(health_pct_mod, px_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (full_df with pct columns, simple_df with frame/pcts)."""
    px_df = px_df.reset_index(drop=True)

    p1_pct, p1_mono = health_pct_mod._process_side(px_df, "p1")
    p2_pct, p2_mono = health_pct_mod._process_side(px_df, "p2")

    full_df = px_df.copy()
    full_df["p1_pct"] = pd.to_numeric(p1_pct, errors="coerce")
    full_df["p1_pct_mono"] = pd.to_numeric(p1_mono, errors="coerce")
    full_df["p2_pct"] = pd.to_numeric(p2_pct, errors="coerce")
    full_df["p2_pct_mono"] = pd.to_numeric(p2_mono, errors="coerce")

    frame_col = health_pct_mod._find_frame_col(full_df)
    if frame_col is not None:
        base = pd.to_numeric(full_df[frame_col], errors="coerce")
        first_valid = base.dropna().iloc[0] if base.notna().any() else 0
        simple_frame = (base - first_valid + 1).astype("Int64")
    else:
        simple_frame = pd.Series(range(1, len(full_df) + 1), dtype="Int64")

    simple = pd.DataFrame(
        {
            "frame": simple_frame.astype("object"),
            "p1_pct": pd.to_numeric(p1_mono, errors="coerce"),
            "p2_pct": pd.to_numeric(p2_mono, errors="coerce"),
        }
    )

    for col in ("p1_pct", "p2_pct"):
        simple[col] = simple[col].round(0).astype("Int64").astype("object")
        simple.loc[pd.isna(simple[col]), col] = ""

    simple.loc[pd.isna(simple["frame"]), "frame"] = ""
    return full_df, simple


# --------------------------------------------------------------------------- #
# Match flow scoring helpers
# --------------------------------------------------------------------------- #


def _normalize_pct_col(series: pd.Series) -> pd.Series:
    series = series.astype(float)
    return (series.clip(0, 100) / 100.0) if series.max() > 1.5 else series.clip(0, 1)


def _opponent_side(side: str) -> str:
    return "P2" if str(side).upper() == "P1" else "P1"


def _format_score(score: float) -> str:
    rounded = round(score, 2)
    mag_str = f"{abs(rounded):.2f}".rstrip("0").rstrip(".")
    if not mag_str:
        mag_str = "0"
    sign = "-" if rounded < 0 else ""
    return f"{sign}{mag_str}"


def load_health_csv(path: Path) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(path, engine="python")
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols

    p1c = next((c for c in cols if ("p1" in c and "pct" in c)), None)
    p2c = next((c for c in cols if ("p2" in c and "pct" in c)), None)

    if p1c and p2c:
        p1 = _normalize_pct_col(df[p1c])
        p2 = _normalize_pct_col(df[p2c])
    else:
        p1c = next((c for c in cols if ("p1" in c and "px" in c)), None)
        p2c = next((c for c in cols if ("p2" in c and "px" in c)), None)
        if not (p1c and p2c):
            raise ValueError("Could not find p1/p2 pct or px columns in the CSV.")
        p1_raw = df[p1c].astype(float).clip(lower=0)
        p2_raw = df[p2c].astype(float).clip(lower=0)
        p1 = (p1_raw / max(p1_raw.max(), 1.0)).clip(0, 1)
        p2 = (p2_raw / max(p2_raw.max(), 1.0)).clip(0, 1)

    p1 = p1.ffill().bfill()
    p2 = p2.ffill().bfill()
    return df, p1.reset_index(drop=True), p2.reset_index(drop=True)


def _infer_duration_frames(df: pd.DataFrame) -> int:
    candidates = ["frame", "frame_idx", "frame_number", "frameid", "frames"]
    for cand in candidates:
        if cand in df.columns:
            series = pd.to_numeric(df[cand], errors="coerce").dropna()
            if not series.empty:
                first = int(series.iloc[0])
                last = int(series.iloc[-1])
                if last >= first:
                    return int(last - first + 1)
    return int(len(df))


def parse_side_result_round(name: str) -> Tuple[str, str, Optional[int]]:
    match = re.search(r"_(P[12])_([WL])_", name, flags=re.IGNORECASE)
    side = match.group(1).upper() if match else "P1"
    result = match.group(2).upper() if match else "W"
    round_idx: Optional[int] = None
    match_round = re.search(r"_(R([123]))_", name, flags=re.IGNORECASE)
    if match_round:
        round_idx = int(match_round.group(2))
    else:
        match_round_alt = re.search(r"_round[-_]?([123])", name, flags=re.IGNORECASE)
        if match_round_alt:
            round_idx = int(match_round_alt.group(1))
    return side, result, round_idx


def _drop_indices(health: pd.Series) -> List[int]:
    diffs = health.diff().fillna(0.0)
    return list(diffs[diffs < 0].index.to_list())


def longest_unharmed_streak_by_events(health: pd.Series) -> int:
    drop_idx = _drop_indices(health)
    total = len(health)
    if total == 0:
        return 0
    if not drop_idx:
        return total
    gaps: List[int] = [drop_idx[0] - 0]
    for a, b in zip(drop_idx, drop_idx[1:]):
        gaps.append(b - a)
    gaps.append((total - 1) - drop_idx[-1])
    return max(gaps) if gaps else 0


def count_damage_events(health: pd.Series) -> int:
    return len(_drop_indices(health))


def validate_weights(weights: Dict[str, float], ranges: Dict[str, Tuple[float, float]], enforce_sum: bool = False) -> Dict[str, float]:
    clamped: Dict[str, float] = {}
    for key, value in weights.items():
        lo, hi = ranges.get(key, (0.0, 1.0))
        clamped[key] = float(min(max(value, lo), hi))
    if enforce_sum:
        total = sum(clamped.values())
        if total > 0:
            clamped = {k: v / total for k, v in clamped.items()}
    return clamped


def match_flow_score_from_series(
    p_health: pd.Series,
    o_health: pd.Series,
    duration_frames: int,
    winner_sign: int,
    T_max: int,
    w_duration: float,
    w_health: float,
    w_streak: float,
    w_tempo: float,
) -> Tuple[float, Dict[str, float]]:
    total_samples = int(len(p_health))
    duration_ratio = min(duration_frames, T_max) / float(max(T_max, 1))
    duration_factor = (1.0 - duration_ratio) ** 2
    duration_term = winner_sign * w_duration * duration_factor

    final_health_margin = float(p_health.iloc[-1] - o_health.iloc[-1]) if total_samples > 0 else 0.0
    health_term = w_health * (final_health_margin ** 2) * (1 if final_health_margin >= 0 else -1)

    longest_self = longest_unharmed_streak_by_events(p_health)
    longest_opp = longest_unharmed_streak_by_events(o_health)
    streak_term = winner_sign * w_streak * (
        (longest_self / max(total_samples, 1)) - (longest_opp / max(total_samples, 1))
    )

    events_self = count_damage_events(p_health)
    events_opp = count_damage_events(o_health)
    denom = events_self + events_opp
    pressure_balance = ((events_opp - events_self) / denom) if denom > 0 else 0.0
    tempo_term = w_tempo * pressure_balance

    score = 100.0 * (duration_term + health_term + streak_term + tempo_term)
    score = max(min(score, 100.0), -100.0)

    diagnostics = {
        "frames_T": duration_frames,
        "sample_rows": total_samples,
        "duration_factor": duration_factor,
        "duration_term_x100": duration_term * 100.0,
        "final_health_margin": final_health_margin,
        "Lp_frames": longest_self,
        "Lo_frames": longest_opp,
        "longest_safe_frac_player": longest_self / max(total_samples, 1),
        "longest_safe_frac_opponent": longest_opp / max(total_samples, 1),
        "streak_term_x100": streak_term * 100.0,
        "events_player_loss": events_self,
        "events_opponent_loss": events_opp,
        "pressure_balance": pressure_balance,
        "tempo_term_x100": tempo_term * 100.0,
        "winner_sign": winner_sign,
    }
    return score, diagnostics


# --------------------------------------------------------------------------- #
# Match flow report aggregator
# --------------------------------------------------------------------------- #


def run_match_flow_report(source_dir: Path, recursive: bool = True, settings: Dict[str, object] | None = None) -> Optional[Path]:
    """
    Aggregate per-round pct CSVs under `source_dir` into a consolidated report.
    Returns the path to the generated report, or None if nothing was produced.
    """
    pattern = REPORT_FILE_GLOB
    output_name = "match_flow_report.csv"
    weights = REPORT_WEIGHTS.copy()
    weight_ranges = REPORT_WEIGHT_RANGES
    enforce_sum = REPORT_ENFORCE_SUM_TO_ONE
    t_max = REPORT_T_MAX
    use_pct_winner = REPORT_USE_PCT_WINNER

    if settings:
        pattern = str(settings.get("report_file_glob", pattern))
        output_name = str(settings.get("report_output_name", output_name))
        override_weights = settings.get("report_weights")
        if isinstance(override_weights, dict):
            for key, value in override_weights.items():
                if key in weights:
                    weights[key] = float(value)
        enforce_sum = bool(settings.get("report_enforce_sum_to_one", enforce_sum))
        t_max = int(settings.get("report_t_max", t_max))
        use_pct_winner = bool(settings.get("report_use_pct_winner", use_pct_winner))

    iterator: Iterable[Path]
    if recursive:
        iterator = source_dir.rglob(pattern)
    else:
        iterator = source_dir.glob(pattern)

    files = [p for p in iterator if p.is_file() and "pct" in p.stem.lower()]
    if not files:
        print("[WARN] No pct CSVs found for match-flow scoring.")
        return None

    weights_eff = validate_weights(weights, weight_ranges, enforce_sum)

    output_csv = source_dir / output_name
    meta_csv = output_csv.with_name(output_csv.stem + "_meta.csv")

    rows = []
    for f in files:
        if f.resolve() in {output_csv.resolve(), meta_csv.resolve()}:
            continue
        try:
            df_raw, p1, p2 = load_health_csv(f)
            duration_frames = _infer_duration_frames(df_raw)
            side, result, rnum = parse_side_result_round(f.name)
            winner_side = side if result == "W" else _opponent_side(side)
            loser_side = _opponent_side(winner_side)
            winner_sign = +1 if result == "W" else -1

            if use_pct_winner:
                final_p1 = float(p1.iloc[-1]) if not p1.empty else float("nan")
                final_p2 = float(p2.iloc[-1]) if not p2.empty else float("nan")
                if pd.notna(final_p1) and pd.notna(final_p2) and final_p1 != final_p2:
                    side = "P1" if final_p1 > final_p2 else "P2"
                    result = "W"
                    winner_side = side
                    loser_side = _opponent_side(side)
                    winner_sign = +1

            player_series, opp_series = (p1, p2) if side == "P1" else (p2, p1)
            score, diag = match_flow_score_from_series(
                player_series,
                opp_series,
                duration_frames,
                winner_sign,
                T_max=t_max,
                w_duration=weights_eff["w_duration"],
                w_health=weights_eff["w_health"],
                w_streak=weights_eff["w_streak"],
                w_tempo=weights_eff["w_tempo"],
            )
            score_display = _format_score(score)

            rows.append(
                {
                    "file": str(f.relative_to(source_dir)),
                    "round": rnum,
                    "side": side,
                    "result": result,
                    "frames_T": int(diag["frames_T"]),
                    "sample_rows": int(diag["sample_rows"]),
                    "duration_factor": round(diag["duration_factor"], 6),
                    "final_health_margin": round(diag["final_health_margin"], 6),
                    "Lp_frames": diag["Lp_frames"],
                    "Lo_frames": diag["Lo_frames"],
                    "longest_safe_frac_player": round(diag["longest_safe_frac_player"], 6),
                    "longest_safe_frac_opponent": round(diag["longest_safe_frac_opponent"], 6),
                    "events_player_loss": diag["events_player_loss"],
                    "events_opponent_loss": diag["events_opponent_loss"],
                    "pressure_balance": round(diag["pressure_balance"], 6),
                    "MatchFlowScore": score_display,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "file": str(f.relative_to(source_dir)),
                    "round": None,
                    "side": None,
                    "result": None,
                    "frames_T": None,
                    "sample_rows": None,
                    "duration_factor": None,
                    "final_health_margin": None,
                    "Lp_frames": None,
                    "Lo_frames": None,
                    "longest_safe_frac_player": None,
                    "longest_safe_frac_opponent": None,
                    "events_player_loss": None,
                    "events_opponent_loss": None,
                    "pressure_balance": None,
                    "MatchFlowScore": None,
                    "error": repr(exc),
                }
            )

    if not rows:
        print("[WARN] Match-flow scorer found files but produced no rows.")
        return None

    report_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_csv, index=False)

    meta = pd.DataFrame(
        [
            {
                "T_MAX": t_max,
                "w_duration": weights_eff["w_duration"],
                "w_health": weights_eff["w_health"],
                "w_streak": weights_eff["w_streak"],
                "w_tempo": weights_eff["w_tempo"],
                "weights_sum": sum(weights_eff.values()),
                "ENFORCE_SUM_TO_ONE": enforce_sum,
                "source_dir": str(source_dir),
                "pattern": pattern,
                "recursive": recursive,
                "score_name": "MatchFlowScore",
            }
        ]
    )
    meta.to_csv(meta_csv, index=False)
    print(f"[OK] Match-flow report saved to: {output_csv}")
    print(f"[OK] Meta saved to: {meta_csv}")
    return output_csv


# --------------------------------------------------------------------------- #
# Video processing orchestration
# --------------------------------------------------------------------------- #


@dataclass
class ProcessingConfig:
    device: torch.device
    model_path: Path
    fight_model: torch.nn.Module
    fight_transform: object
    fight_mod: object
    round_end_finder: object
    health_px_mod: object
    health_pct_mod: object
    round1_start: int
    round2_start: int
    round3_offset_after_r2: int
    round3_early_fallback: int
    max_rounds: int
    side_option: str
    max_round_frames: int
    min_gap: int
    sample_step: int
    output_dir: Path


def process_video(video_path: Path, cfg: ProcessingConfig) -> Dict:
    total_cap = cv2.VideoCapture(str(video_path))
    total_frames = None
    if total_cap.isOpened():
        total_frames = int(total_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_cap.release()

    print(f"[INFO] >>> Starting video '{video_path.name}' ({total_frames or 'unknown'} frames)")
    rounds: List[Dict] = []
    search_start = cfg.round1_start
    last_fight_frame: Optional[int] = None

    for round_idx in range(1, cfg.max_rounds + 1):
        if round_idx == 1:
            round_seed = cfg.round1_start
        elif round_idx == 2:
            round_seed = max(search_start, cfg.round2_start)
        elif round_idx == 3 and last_fight_frame is not None:
            round_seed = max(search_start, last_fight_frame + cfg.round3_offset_after_r2)
        else:
            round_seed = search_start

        if total_frames is not None and round_seed >= total_frames:
            break

        print(f"[INFO]     Round {round_idx}: searching fight frame from {round_seed}")
        fight_result = cfg.fight_mod.infer_video_optimized(
            str(video_path),
            cfg.fight_model,
            cfg.fight_transform,
            start_frame=round_seed,
            device=str(cfg.device),
            use_adjacent_confirm=getattr(cfg.fight_mod, "ADJACENT_CONFIRM_ENABLED", False),
            confirm_direction=getattr(cfg.fight_mod, "ADJACENT_CONFIRM_DIRECTION", "next"),
            confirm_n=getattr(cfg.fight_mod, "ADJACENT_CONFIRM_N_FRAMES", 1),
            confirm_threshold=getattr(cfg.fight_mod, "ADJACENT_CONFIRM_CONFIDENCE", 0.98),
            confirm_strategy=getattr(cfg.fight_mod, "ADJACENT_CONFIRM_STRATEGY", "any"),
        )
        if not fight_result and round_idx == 3 and cfg.round3_early_fallback > 0:
            fallback_start = max(search_start, round_seed - cfg.round3_early_fallback)
            if fallback_start < round_seed:
                print(f"[INFO]     Round {round_idx}: primary search missed, retrying from {fallback_start} (fallback)")
                fight_result = cfg.fight_mod.infer_video_optimized(
                    str(video_path),
                    cfg.fight_model,
                    cfg.fight_transform,
                    start_frame=fallback_start,
                    device=str(cfg.device),
                    use_adjacent_confirm=getattr(cfg.fight_mod, "ADJACENT_CONFIRM_ENABLED", False),
                    confirm_direction=getattr(cfg.fight_mod, "ADJACENT_CONFIRM_DIRECTION", "next"),
                    confirm_n=getattr(cfg.fight_mod, "ADJACENT_CONFIRM_N_FRAMES", 1),
                    confirm_threshold=getattr(cfg.fight_mod, "ADJACENT_CONFIRM_CONFIDENCE", 0.98),
                    confirm_strategy=getattr(cfg.fight_mod, "ADJACENT_CONFIRM_STRATEGY", "any"),
                )
        if not fight_result:
            print(f"[WARN]     Round {round_idx}: fight frame not found; stopping video early.")
            break

        start_frame = int(fight_result[0])
        last_fight_frame = start_frame
        fight_conf = float(fight_result[1]) if len(fight_result) > 1 else None
        print(f"[INFO]     Round {round_idx}: fight frame @ {start_frame} (conf={fight_conf})")

        end_frame, resolved_side = cfg.round_end_finder.find_round_end(
            video_path=video_path,
            start_frame=start_frame,
            max_frames=cfg.max_round_frames,
            side=cfg.side_option,
        )
        if end_frame is None:
            print(f"[WARN]     Round {round_idx}: round end not detected; aborting remaining rounds.")
            break
        print(f"[INFO]     Round {round_idx}: round end @ {end_frame} using side {resolved_side}")

        rows = extract_health_px_rows(
            cfg.health_px_mod,
            video_path=video_path,
            video_basename=video_path.stem,
            round_id=round_idx,
            start_frame=start_frame,
            end_frame=end_frame,
            sample_step=cfg.sample_step,
        )
        if not rows:
            print(f"[WARN]     Round {round_idx}: health extraction returned no rows.")
            break

        px_df = pd.DataFrame(rows)
        cfg.health_pct_mod.SIDE_WIDTH["p1"] = cfg.health_px_mod.P1_WIDTH
        cfg.health_pct_mod.SIDE_WIDTH["p2"] = cfg.health_px_mod.P2_WIDTH
        full_df, simple_df = build_pct_outputs(cfg.health_pct_mod, px_df)

        round_dir = cfg.output_dir / video_path.stem
        round_dir.mkdir(parents=True, exist_ok=True)

        px_filename = f"{video_path.stem}__round{round_idx}_health_px.csv"
        pct_filename = px_filename.replace(
            "_health_px.csv",
            cfg.health_pct_mod.SIMPLE_FILENAME_SUFFIX,
        )

        full_path = round_dir / px_filename
        pct_path = round_dir / pct_filename

        print(f"[INFO]     Round {round_idx}: writing px -> {full_path}")
        print(f"[INFO]     Round {round_idx}: writing pct -> {pct_path}")
        full_df.to_csv(full_path, index=False, encoding="utf-8-sig")
        simple_df.to_csv(pct_path, index=False, encoding="utf-8")

        rounds.append(
            {
                "round": round_idx,
                "fight_frame": start_frame,
                "fight_confidence": fight_conf,
                "round_end_frame": end_frame,
                "duration_frames": int(end_frame - start_frame + 1),
                "side_used_for_end": resolved_side,
                "px_csv": str(full_path),
                "pct_csv": str(pct_path),
                "samples": len(full_df),
            }
        )

        search_start = end_frame + cfg.min_gap

    summary = {
        "video": str(video_path),
        "total_frames": total_frames,
        "rounds_processed": rounds,
    }
    if rounds:
        summary_path = cfg.output_dir / video_path.stem / "summary.json"
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        summary["summary_json"] = str(summary_path)
        print(f"[INFO] >>> Finished video '{video_path.name}' with {len(rounds)} round(s). Summary: {summary_path}")
    else:
        print(f"[WARN] >>> No rounds processed for '{video_path.name}'.")
    return summary


def gather_videos(path: Path, extensions: Iterable[str], recursive: bool) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")
    pattern = "**/*" if recursive else "*"
    files = [
        p
        for p in path.glob(pattern)
        if p.is_file() and p.suffix.lower() in extensions
    ]
    return sorted(files)


def execute_pipeline(settings: Dict[str, object], progress_callback: Optional[Callable[[int, int, Optional[Path]], None]] = None) -> Optional[Path]:
    script_dir = Path(__file__).resolve().parent

    def _resolve_relative(path_like: object | None) -> Optional[Path]:
        if path_like is None:
            return None
        candidate = Path(path_like)
        return candidate if candidate.is_absolute() else script_dir / candidate

    print("[INFO] Loading helper modules...")
    modules = ProjectModules.load(script_dir)
    print("[INFO] Modules ready.")

    user_cfg = settings.copy()

    input_path_raw = user_cfg.get("input_path")
    if input_path_raw is None:
        raise ValueError("SETTINGS['input_path'] must be set to a video file or folder.")
    input_path = Path(input_path_raw)

    device_override = user_cfg.get("device")
    device = torch.device(device_override if device_override else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Using device: {device}")

    model_path_raw = user_cfg.get("model_path")
    if model_path_raw:
        model_path = _resolve_relative(model_path_raw)
    else:
        module_model_path = Path(getattr(modules.fight, "MODEL_PATH", script_dir / "model.pth"))
        if not module_model_path.is_absolute():
            module_model_path = Path(getattr(modules.fight, "__file__", script_dir)).resolve().parent / module_model_path
        model_path = module_model_path
    print(f"[INFO] Loading fight-frame model from: {model_path}")
    fight_model, fight_transform = _create_fight_model(modules.fight, model_path, device)
    print("[INFO] Fight-frame model ready.")

    round_end_tail = int(user_cfg.get("round_end_tail", modules.round_end.DEFAULT_TAIL_CONSECUTIVE_NONE))
    round_end_debug_dir_raw = user_cfg.get("round_end_debug_dir", None)
    round_end_debug_dir = _resolve_relative(round_end_debug_dir_raw) if round_end_debug_dir_raw else None
    round_end_finder = modules.round_end.RoundEndFinder(
        tail_consecutive_none=round_end_tail,
        debug_dir=round_end_debug_dir,
    )
    print(f"[INFO] Round-end finder configured (tail={round_end_tail}, coarse_step={round_end_finder.coarse_step}).")
    configure_health_px_module(modules.health_px)

    modules.health_pct.P1_WIDTH = modules.health_px.P1_WIDTH
    modules.health_pct.P2_WIDTH = modules.health_px.P2_WIDTH
    modules.health_pct.SIDE_WIDTH["p1"] = modules.health_px.P1_WIDTH
    modules.health_pct.SIDE_WIDTH["p2"] = modules.health_px.P2_WIDTH

    output_dir_raw = user_cfg.get("output_dir", Path("output"))
    output_dir = _resolve_relative(output_dir_raw)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions_raw = user_cfg.get("extensions", (".mp4", ".mkv", ".avi", ".mov"))
    if isinstance(extensions_raw, str):
        extensions = tuple(ext.strip().lower() for ext in extensions_raw.split(",") if ext.strip())
    else:
        extensions = tuple(str(ext).lower() for ext in extensions_raw)

    recursive_search = bool(user_cfg.get("recursive", False))
    videos = gather_videos(input_path, extensions, recursive_search)
    if not videos:
        raise FileNotFoundError("No videos found matching the provided criteria.")
    print(f"[INFO] Videos discovered: {len(videos)} (recursive={recursive_search})")

    def _resolve_round_setting(name: str, module_attr: str, fallback: int) -> int:
        value = user_cfg.get(name)
        if value is None:
            value = getattr(modules.fight, module_attr, fallback)
        if value is None:
            value = fallback
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"SETTINGS['{name}'] must be an integer-compatible value (got {value!r}).") from exc

    round1_start = _resolve_round_setting("round1_start", "ROUND1_START", 0)
    round2_start = _resolve_round_setting("round2_start", "ROUND2_START", round1_start)
    round3_offset_after_r2 = _resolve_round_setting("round3_offset_after_r2", "ROUND3_OFFSET_AFTER_R2", 0)
    round3_early_fallback = _resolve_round_setting("round3_early_fallback", "ROUND3_EARLY_FALLBACK", 0)

    cfg = ProcessingConfig(
        device=device,
        model_path=model_path,
        fight_model=fight_model,
        fight_transform=fight_transform,
        fight_mod=modules.fight,
        round_end_finder=round_end_finder,
        health_px_mod=modules.health_px,
        health_pct_mod=modules.health_pct,
        round1_start=round1_start,
        round2_start=round2_start,
        round3_offset_after_r2=round3_offset_after_r2,
        round3_early_fallback=round3_early_fallback,
        max_rounds=int(user_cfg.get("max_rounds", 3)),
        side_option=str(user_cfg.get("side", "auto")),
        max_round_frames=int(user_cfg.get("max_round_frames", 7200)),
        min_gap=max(1, int(user_cfg.get("min_gap", 480))),
        sample_step=max(1, int(user_cfg.get("sample_step", 1))),
        output_dir=output_dir,
    )

    summaries = []
    if progress_callback:
        progress_callback(0, len(videos), None)
    for idx, video in enumerate(videos, start=1):
        print(f"[{idx}/{len(videos)}] Processing {video}")
        if progress_callback:
            progress_callback(idx - 1, len(videos), video)
        try:
            summary = process_video(video, cfg)
            summaries.append(summary)
        except Exception as exc:
            print(f"[ERROR] Failed on {video}: {exc}")
        if progress_callback:
            progress_callback(idx, len(videos), video)

    summary_path = output_dir / "batch_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2)
    print(f"[DONE] Processed {len(videos)} video(s). Batch summary: {summary_path}")

    print("[INFO] Building match-flow score report...")
    report_path = run_match_flow_report(output_dir, recursive=True, settings=settings)
    print("[INFO] Match flow pipeline completed.")
    return report_path


def launch_gui():
    root = tk.Tk()
    root.title("Match Flow Score Tool")

    path_var = tk.StringVar()
    status_var = tk.StringVar(value="Idle")

    ttk.Label(root, text="Path:").grid(row=0, column=0, sticky="w", padx=8, pady=(10, 4))
    path_entry = ttk.Entry(root, textvariable=path_var, width=50)
    path_entry.grid(row=0, column=1, sticky="ew", padx=8, pady=(10, 4))

    selection_var = tk.StringVar(value="No input selected.")

    def load_file():
        filetypes = [("Video files", "*.mp4 *.mkv *.avi *.mov"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Select video file", filetypes=filetypes)
        if path:
            path_var.set(path)
            selection_var.set(f"Selected file: {Path(path).name}")

    def load_folder():
        path = filedialog.askdirectory(title="Select folder with videos")
        if path:
            path_var.set(path)
            selection_var.set(f"Selected folder: {Path(path).name}")

    button_frame = ttk.Frame(root)
    button_frame.grid(row=0, column=2, rowspan=2, padx=8, pady=(8, 4), sticky="nsew")
    load_file_btn = ttk.Button(button_frame, text="Load File", command=load_file)
    load_file_btn.pack(fill="x", pady=(2, 2))
    load_folder_btn = ttk.Button(button_frame, text="Load Folder", command=load_folder)
    load_folder_btn.pack(fill="x", pady=(2, 2))

    selection_label = ttk.Label(root, textvariable=selection_var)
    selection_label.grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 4))

    status_label = ttk.Label(root, textvariable=status_var)
    status_label.grid(row=2, column=0, columnspan=3, sticky="w", padx=8, pady=(4, 8))

    animation_state = {"job": None, "idx": 0, "base": "Processing"}
    dots_cycle = ["", ".", "..", "..."]

    def _animate_status():
        animation_state["idx"] = (animation_state["idx"] + 1) % len(dots_cycle)
        status_var.set(f"{animation_state['base']}{dots_cycle[animation_state['idx']]}")
        animation_state["job"] = root.after(500, _animate_status)

    def start_animation(message: str):
        animation_state["base"] = message
        animation_state["idx"] = 0
        status_var.set(message)
        if animation_state["job"] is None:
            animation_state["job"] = root.after(500, _animate_status)

    def stop_animation(message: str):
        job = animation_state.get("job")
        if job is not None:
            root.after_cancel(job)
            animation_state["job"] = None
        status_var.set(message)

    def set_running(running: bool):
        state = "disabled" if running else "normal"
        load_file_btn.config(state=state)
        load_folder_btn.config(state=state)
        start_btn.config(state=state)

    def update_progress(completed: int, total: int, current_video: Optional[Path]):
        if total == 0:
            start_animation("Processing")
            return
        if current_video is None:
            start_animation("Scanning input")
        else:
            start_animation(f"Processing {current_video.name} ({completed}/{total})")

    def show_report_window(report_path: Optional[Path]):
        if report_path is None or not report_path.exists():
            messagebox.showinfo("Match Flow Score", "Processing completed, but no report was produced.")
            return
        try:
            df = pd.read_csv(report_path)
        except Exception as exc:
            messagebox.showerror("Match Flow Score", f"Could not read report: {exc}")
            return

        win = tk.Toplevel(root)
        win.title("Match Flow Scores")

        columns = []
        for col in ("file", "MatchFlowScore"):
            if col in df.columns:
                columns.append(col)
        if not columns and not df.empty:
            columns = list(df.columns[:4])
        if not columns:
            messagebox.showinfo("Match Flow Score", "Report is empty.")
            win.destroy()
            return

        tree = ttk.Treeview(win, columns=columns, show="headings")
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=200 if col == "file" else 120, anchor="center")

        for _, row in df.iterrows():
            values = [row.get(col, "") for col in columns]
            tree.insert("", "end", values=values)

        tree.pack(fill="both", expand=True)

        ttk.Button(win, text="Close", command=win.destroy).pack(pady=6)

    def finish(success: bool, report_path: Optional[Path], message: str = ""):
        set_running(False)
        stop_animation("Completed." if success else "Failed.")
        if success:
            show_report_window(report_path)
        else:
            messagebox.showerror("Match Flow Score", message or "Processing failed.")

    def start_processing():
        path = path_var.get().strip()
        if not path:
            messagebox.showwarning("Match Flow Score", "Please select a file or folder.")
            return
        selected = Path(path)
        if not selected.exists():
            messagebox.showwarning("Match Flow Score", "Selected path does not exist.")
            return
        if not (selected.is_file() or selected.is_dir()):
            messagebox.showwarning("Match Flow Score", "Please choose a video file or a folder containing videos.")
            return

        set_running(True)
        start_animation("Starting")

        def worker():
            try:
                settings = SETTINGS.copy()
                settings["input_path"] = Path(path)
                report_path = execute_pipeline(settings, lambda c, t, v: root.after(0, update_progress, c, t, v))
                root.after(0, finish, True, report_path, "")
            except Exception as exc:
                root.after(0, finish, False, None, str(exc))

        threading.Thread(target=worker, daemon=True).start()

    start_btn = ttk.Button(root, text="Start Processing", command=start_processing)
    start_btn.grid(row=4, column=0, columnspan=3, pady=(0, 12))

    root.columnconfigure(1, weight=1)
    root.minsize(520, 180)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
