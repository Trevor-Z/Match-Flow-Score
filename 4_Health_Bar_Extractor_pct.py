from pathlib import Path
import pandas as pd
import numpy as np
import math

# ===================== CONFIG =====================

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR  = SCRIPT_DIR / "output" / "px"
OUTPUT_DIR = SCRIPT_DIR / "output" / "pct"
GLOB       = "*_health_px.csv"

# ROI widths (must match extractor ROIs)
P1_WIDTH = 670
P2_WIDTH = 670
SIDE_WIDTH = {"p1": P1_WIDTH, "p2": P2_WIDTH}

# Quantization & flow
QUANTIZE_PERCENT = 1          # round to nearest %
CONFIRM_N        = 3          # need N consecutive accepted rows with same quantized value

# Px guard: require movement vs the px recorded while current displayed % was active
MIN_PX_DELTA_TO_CHANGE = 6

# Change-confirmation stability
STABILITY_LOOKAHEAD_SAMPLES        = 0
STABILITY_REQUIRED_MATCHES         = 1
STABILITY_MAX_DELTA_PCT            = 2.0
STABILITY_ACCEPT_ON_SHORT_LOOKAHEAD = True

# AAR / ARR strict matching for rejected rows
REJECT_MATCH_TOL_PCT = 1.0
REQUIRE_BOTH_IN_ARR  = True

# Minimum non-zero floor for mono percent when px tiny but nonzero
MIN_NONZERO_MONO_PX  = 4
MIN_NONZERO_MONO_PCT = 1

# Output controls
LINE_TERM = "\n"

# ---- SIMPLE OUTPUT like *_health_pct_x.csv ----
SIMPLE_OUTPUT_MODE = True    # <- enable simple CSV
SIMPLE_FILENAME_SUFFIX = "_health_pct.csv"

# Accept final row values as-is (per side) without running confirmation logic
ZERO_LAST_ROW_IF_ZERO = True
# ===================================================


def _quantize(p):
    if p is None or pd.isna(p): return np.nan
    s = float(QUANTIZE_PERCENT)
    if s <= 0: return round(float(p), 2)
    return round(s * round(float(p)/s), 2)

def _insert_right_of(df: pd.DataFrame, right_of_col: str, new_col: str, values):
    if right_of_col in df.columns:
        pos = df.columns.get_loc(right_of_col) + 1
        df.insert(pos, new_col, values)
    else:
        df[new_col] = values

def _pct_from_px(px, width):
    try:
        pxf = float(px); wf = float(width)
        return (100.0 * pxf / wf) if (wf > 0 and pxf > 0) else np.nan
    except Exception:
        return np.nan

def _guard_switch(q_prev, px_prev, q_cand, px_now):
    if q_cand is None or (isinstance(q_cand,float) and math.isnan(q_cand)):
        return q_prev, px_prev, False
    if q_prev is None or px_prev is None or (isinstance(px_prev,float) and math.isnan(px_prev)):
        return q_cand, float(px_now), True
    if q_cand == q_prev:
        return q_prev, float(px_now), False
    try:
        px_now_f = float(px_now)
        px_prev_f = float(px_prev)
    except Exception:
        return q_cand, float(px_now), True
    if abs(px_now_f - px_prev_f) >= float(MIN_PX_DELTA_TO_CHANGE):
        return q_cand, px_now_f, True
    else:
        return q_prev, px_prev, False

def _rej_matches_candidate(px_val, width, cand_q) -> bool:
    rp = _pct_from_px(px_val, width)
    if pd.isna(rp):
        return False
    rq = _quantize(rp)
    if pd.isna(rq) or pd.isna(cand_q):
        return False
    return abs(float(rq) - float(cand_q)) <= float(REJECT_MATCH_TOL_PCT)

def _future_agree_with_current(raw_series, start_idx_exclusive, K, current_raw, tol):
    agree = 0
    avail = 0
    n = len(raw_series)
    j = start_idx_exclusive + 1
    while j < n and avail < K:
        v = raw_series.iat[j]
        if pd.notna(v):
            avail += 1
            if abs(v - current_raw) <= tol:
                agree += 1
        j += 1
    return agree, avail

def _mono_floor_if_tiny(q_val, px_val):
    try:
        if pd.isna(q_val):
            return q_val
        qf = float(q_val)
        pxf = float(px_val)
        if (pxf > 0) and (pxf < float(MIN_NONZERO_MONO_PX)) and (qf <= 0.0):
            return float(MIN_NONZERO_MONO_PCT)
    except Exception:
        pass
    return q_val

def _find_frame_col(df: pd.DataFrame) -> str | None:
    # Try common spellings, case-insensitive
    candidates = ["frame", "frame_idx", "frame_number", "frameid", "frm"]
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    # If exact not found, try any column whose name contains "frame"
    for c in df.columns:
        if "frame" in c.lower():
            return c
    return None

def _process_side(df: pd.DataFrame, side: str):
    status_col = f"{side}_status"; px_col = f"{side}_px"
    if status_col not in df or px_col not in df:
        n = len(df)
        return pd.Series([np.nan]*n, dtype="float64"), pd.Series([np.nan]*n, dtype="float64")

    n = len(df)
    width = SIDE_WIDTH[side]

    # raw pct from px for ACCEPTED rows; NaN for rejected
    raw_pct = []
    for i in range(n):
        if str(df.at[i, status_col]).lower() == "accepted":
            raw_pct.append(_pct_from_px(df.at[i, px_col], width))
        else:
            raw_pct.append(np.nan)
    raw_pct = pd.Series(raw_pct, dtype="float64")

    out_pct  = pd.Series([np.nan]*n, dtype="float64")
    out_mono = pd.Series([100.0]*n, dtype="float64")  # monotonic floor, start 100

    # state
    last_display_q = None
    last_px_for_current_q = None
    last_mono = 100.0

    # candidate run (confirm-N)
    cand_q = None
    cand_start_idx = None
    cand_count = 0
    started_with_accepted = False

    i = 0
    while i < n:
        out_pct.iat[i]  = last_display_q if last_display_q is not None else np.nan
        out_mono.iat[i] = last_mono

        st = str(df.at[i, status_col]).lower()
        rp = raw_pct.iat[i]
        px_now = df.at[i, px_col]

        if st == "accepted" and pd.notna(rp):
            q = _quantize(rp)

            if cand_q is None or q != cand_q:
                cand_q = q
                cand_start_idx = i
                cand_count = 1
                started_with_accepted = True
            else:
                cand_count += 1
                if cand_count >= CONFIRM_N:
                    q_out, px_out, switched_by_px = _guard_switch(last_display_q, last_px_for_current_q, q, px_now)

                    change_requested = (last_display_q is None) or (q != last_display_q)
                    change_ok = True
                    if change_requested:
                        if last_display_q is not None:
                            agree, avail = _future_agree_with_current(
                                raw_series=raw_pct,
                                start_idx_exclusive=i,
                                K=STABILITY_LOOKAHEAD_SAMPLES,
                                current_raw=rp,
                                tol=STABILITY_MAX_DELTA_PCT
                            )
                            if avail == 0 and STABILITY_ACCEPT_ON_SHORT_LOOKAHEAD:
                                change_ok = True
                            else:
                                change_ok = (agree >= STABILITY_REQUIRED_MATCHES)
                        else:
                            change_ok = True

                    if (switched_by_px or last_display_q is None) and change_ok:
                        last_display_q = q_out
                        last_px_for_current_q = px_out
                        out_pct.iloc[cand_start_idx:i+1] = last_display_q

                        commit_px = px_now
                        mono_commit = _mono_floor_if_tiny(q_out, commit_px)
                        last_mono = min(last_mono, float(mono_commit))
                        out_mono.iloc[cand_start_idx:i+1] = last_mono

                        cand_q = None; cand_start_idx = None; cand_count = 0; started_with_accepted = False

        elif st == "rejected":
            if cand_q is not None and started_with_accepted:
                if cand_count == 2 and (i-1) >= 0:
                    if _rej_matches_candidate(df.at[i, px_col], width, cand_q):
                        q_out, px_out, switched = _guard_switch(last_display_q, last_px_for_current_q, cand_q, df.at[i-1, px_col])
                        change_ok = True
                        if (last_display_q is not None) and (cand_q != last_display_q):
                            raw_current = raw_pct.iat[i-1]
                            agree, avail = _future_agree_with_current(raw_pct, i, STABILITY_LOOKAHEAD_SAMPLES, raw_current, STABILITY_MAX_DELTA_PCT)
                            change_ok = (True if (avail == 0 and STABILITY_ACCEPT_ON_SHORT_LOOKAHEAD) else (agree >= STABILITY_REQUIRED_MATCHES))
                        if (switched or last_display_q is None) and change_ok:
                            last_display_q = q_out
                            last_px_for_current_q = px_out
                            out_pct.iloc[cand_start_idx:i+1] = last_display_q

                            commit_px = df.at[i-1, px_col]
                            mono_commit = _mono_floor_if_tiny(q_out, commit_px)
                            last_mono = min(last_mono, float(mono_commit))
                            out_mono.iloc[cand_start_idx:i+1] = last_mono

                            cand_q = None; cand_start_idx = None; cand_count = 0; started_with_accepted = False

                elif cand_count == 1 and (i+1) < n and str(df.at[i+1, status_col]).lower() == "rejected":
                    first_ok  = _rej_matches_candidate(df.at[i,   px_col], width, cand_q)
                    second_ok = _rej_matches_candidate(df.at[i+1, px_col], width, cand_q)
                    allow_arr = (first_ok and second_ok) if REQUIRE_BOTH_IN_ARR else (second_ok or first_ok)
                    if allow_arr:
                        q_out, px_out, switched = _guard_switch(last_display_q, last_px_for_current_q, cand_q, df.at[i+1, px_col])
                        change_ok = True
                        if (last_display_q is not None) and (cand_q != last_display_q):
                            raw_ref_idx = i-1 if (i-1) >= 0 and pd.notna(raw_pct.iat[i-1]) else None
                            raw_current = raw_pct.iat[raw_ref_idx] if raw_ref_idx is not None else np.nan
                            if pd.notna(raw_current):
                                agree, avail = _future_agree_with_current(raw_pct, i+1, STABILITY_LOOKAHEAD_SAMPLES, raw_current, STABILITY_MAX_DELTA_PCT)
                                change_ok = (True if (avail == 0 and STABILITY_ACCEPT_ON_SHORT_LOOKAHEAD) else (agree >= STABILITY_REQUIRED_MATCHES))
                        if (switched or last_display_q is None) and change_ok:
                            last_display_q = q_out
                            last_px_for_current_q = px_out
                            out_pct.iloc[cand_start_idx:i+2] = last_display_q

                            commit_px = df.at[i+1, px_col]
                            mono_commit = _mono_floor_if_tiny(q_out, commit_px)
                            last_mono = min(last_mono, float(mono_commit))
                            out_mono.iloc[cand_start_idx:i+2] = last_mono

                            cand_q = None; cand_start_idx = None; cand_count = 0; started_with_accepted = False
                            i += 1  # consume i+1 as well

            if cand_q is not None:
                cand_q = None; cand_start_idx = None; cand_count = 0; started_with_accepted = False

        i += 1

    if ZERO_LAST_ROW_IF_ZERO and n > 0:
        last_status = str(df.at[n-1, status_col]).lower()
        last_px = df.at[n-1, px_col]
        if last_status == "accepted" and pd.notna(last_px) and float(last_px) == 0.0:
            out_pct.iat[n-1] = 0.0
            out_mono.iat[n-1] = 0.0

    return out_pct, out_mono

# ===================== MAIN =====================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(INPUT_DIR.glob(GLOB))
    if not files:
        print(f"[INFO] No CSVs found in {INPUT_DIR} matching {GLOB}."); return

    for f in files:
        try:
            df = pd.read_csv(f)

            p1_pct, p1_mono = _process_side(df, "p1")
            p2_pct, p2_mono = _process_side(df, "p2")

            # Attach to full DF
            _insert_right_of(df, "p1_px", "p1_pct", p1_pct)
            _insert_right_of(df, "p1_pct", "p1_pct_mono", p1_mono)
            _insert_right_of(df, "p2_px", "p2_pct", p2_pct)
            _insert_right_of(df, "p2_pct", "p2_pct_mono", p2_mono)

            # neat CSV blanks for main mode
            for col in ["p1_pct","p1_pct_mono","p2_pct","p2_pct_mono"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").round(2)
                df[col] = df[col].astype("object")
                df.loc[df[col].isna(), col] = ""

            if SIMPLE_OUTPUT_MODE:
                # --- Simple output like *_health_pct_x.csv ---
                # Frame mapping: start at 1, preserve intervals from source frame column
                frame_col = _find_frame_col(df)
                if frame_col is not None:
                    base = pd.to_numeric(df[frame_col], errors="coerce")
                    first_valid = base.dropna().iloc[0] if base.notna().any() else 0
                    simple_frame = (base - first_valid + 1).astype("Int64")
                else:
                    # Fallback: sequential frames (no interval preservation possible)
                    simple_frame = pd.Series(np.arange(1, len(df)+1), dtype="Int64")

                # Use the monotonic series for simple pcts
                simple = pd.DataFrame({
                    "frame": simple_frame.astype("object"),
                    "p1_pct": pd.to_numeric(p1_mono, errors="coerce"),
                    "p2_pct": pd.to_numeric(p2_mono, errors="coerce"),
                })

                # Round to integer display; blank NaNs
                for c in ["p1_pct","p2_pct"]:
                    simple[c] = simple[c].round(0).astype("Int64").astype("object")
                    simple.loc[pd.isna(simple[c]), c] = ""

                # Also blank any NaNs (rare) in frame
                simple.loc[pd.isna(simple["frame"]), "frame"] = ""

                out_path = OUTPUT_DIR / f.name.replace("_health_px.csv", SIMPLE_FILENAME_SUFFIX)
                simple.to_csv(out_path, index=False, encoding="utf-8", lineterminator=LINE_TERM)
                print(f"[OK] Wrote (simple): {out_path}")

            else:
                # --- Original rich output ---
                out_path = OUTPUT_DIR / f.name.replace("_health_px.csv","_health_pct.csv")
                df.to_csv(out_path, index=False, encoding="utf-8", lineterminator=LINE_TERM)
                print(f"[OK] Wrote: {out_path}")

        except Exception as e:
            print(f"[ERR] {f.name}: {e}")

if __name__ == "__main__":
    main()
