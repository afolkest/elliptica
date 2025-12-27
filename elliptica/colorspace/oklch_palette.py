"""OKLCH palette interpolation and LUT generation."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .gamut import max_chroma_fast, gamut_map_to_srgb


def _coerce_stop(stop: dict) -> dict:
    return {
        "pos": float(stop["pos"]),
        "L": float(stop["L"]),
        "C": float(stop["C"]),
        "H": float(stop["H"]),
    }


def _sorted_stops(stops: Iterable[dict]) -> list[dict]:
    coerced = [_coerce_stop(stop) for stop in stops]
    return sorted(coerced, key=lambda s: s["pos"])


def _chroma_rel_to_abs(L: np.ndarray, H: np.ndarray, C_rel: np.ndarray) -> np.ndarray:
    max_c = max_chroma_fast(L, H)
    return np.clip(C_rel, 0.0, 1.0) * max_c


def _chroma_abs_to_rel(L: np.ndarray, H: np.ndarray, C_abs: np.ndarray) -> np.ndarray:
    max_c = max_chroma_fast(L, H)
    safe = np.where(max_c <= 0.0, 0.0, C_abs / max_c)
    return np.clip(safe, 0.0, 1.0)


def _stop_chroma_rel_abs(stop: dict, relative_chroma: bool) -> tuple[float, float]:
    if relative_chroma:
        c_rel = float(stop["C"])
        c_abs = float(_chroma_rel_to_abs(stop["L"], stop["H"], c_rel))
    else:
        c_abs = float(stop["C"])
        c_rel = float(_chroma_abs_to_rel(stop["L"], stop["H"], c_abs))
    return c_rel, c_abs


def interpolate_oklch(
    stops: Iterable[dict],
    t: float,
    relative_chroma: bool = True,
    interp_mix: float = 1.0,
) -> tuple[float, float, float]:
    """Interpolate OKLCH stops at position t in [0, 1]."""
    stops_sorted = _sorted_stops(stops)
    if not stops_sorted:
        return (0.5, 0.0, 0.0)

    if len(stops_sorted) == 1:
        stop = stops_sorted[0]
        return (stop["L"], stop["C"], stop["H"])

    t = float(np.clip(t, 0.0, 1.0))

    if t <= stops_sorted[0]["pos"]:
        stop = stops_sorted[0]
        return (stop["L"], stop["C"], stop["H"])

    if t >= stops_sorted[-1]["pos"]:
        stop = stops_sorted[-1]
        return (stop["L"], stop["C"], stop["H"])

    mix = float(np.clip(interp_mix, 0.0, 1.0))

    for i in range(len(stops_sorted) - 1):
        s0, s1 = stops_sorted[i], stops_sorted[i + 1]
        if s0["pos"] <= t <= s1["pos"]:
            frac = (t - s0["pos"]) / (s1["pos"] - s0["pos"])

            L = s0["L"] + frac * (s1["L"] - s0["L"])

            h0, h1 = s0["H"], s1["H"]
            dh = h1 - h0
            if dh > 180:
                dh -= 360
            elif dh < -180:
                dh += 360
            H = (h0 + frac * dh) % 360

            c_rel0, c_abs0 = _stop_chroma_rel_abs(s0, relative_chroma)
            c_rel1, c_abs1 = _stop_chroma_rel_abs(s1, relative_chroma)
            c_rel = c_rel0 + frac * (c_rel1 - c_rel0)
            c_abs_rel = _chroma_rel_to_abs(L, H, c_rel)
            c_abs = c_abs0 + frac * (c_abs1 - c_abs0)
            c_abs_mix = (1.0 - mix) * c_abs + mix * c_abs_rel

            if relative_chroma:
                C = float(_chroma_abs_to_rel(L, H, c_abs_mix))
            else:
                C = float(c_abs_mix)

            return (float(L), float(C), float(H))

    stop = stops_sorted[-1]
    return (stop["L"], stop["C"], stop["H"])


def build_oklch_lut(
    stops: Iterable[dict],
    size: int = 256,
    relative_chroma: bool = True,
    interp_mix: float = 1.0,
) -> np.ndarray:
    """Build an RGB LUT from OKLCH stops."""
    stops_sorted = _sorted_stops(stops)
    if size <= 0:
        raise ValueError("LUT size must be positive")

    if not stops_sorted:
        return np.zeros((size, 3), dtype=np.float32)

    positions = np.linspace(0.0, 1.0, size, dtype=np.float32)

    if len(stops_sorted) == 1:
        stop = stops_sorted[0]
        L_arr = np.full(size, stop["L"], dtype=np.float32)
        H_arr = np.full(size, stop["H"], dtype=np.float32)
        c_rel, c_abs = _stop_chroma_rel_abs(stop, relative_chroma)
        C_rel_arr = np.full(size, c_rel, dtype=np.float32)
        C_abs_arr = np.full(size, c_abs, dtype=np.float32)
    else:
        stop_count = len(stops_sorted)
        stop_c_rel = np.empty(stop_count, dtype=np.float32)
        stop_c_abs = np.empty(stop_count, dtype=np.float32)

        for i, stop in enumerate(stops_sorted):
            c_rel, c_abs = _stop_chroma_rel_abs(stop, relative_chroma)
            stop_c_rel[i] = c_rel
            stop_c_abs[i] = c_abs

        L_arr = np.full(size, stops_sorted[0]["L"], dtype=np.float32)
        H_arr = np.full(size, stops_sorted[0]["H"], dtype=np.float32)
        C_rel_arr = np.full(size, stop_c_rel[0], dtype=np.float32)
        C_abs_arr = np.full(size, stop_c_abs[0], dtype=np.float32)

        for i in range(stop_count - 1):
            s0, s1 = stops_sorted[i], stops_sorted[i + 1]
            p0, p1 = float(s0["pos"]), float(s1["pos"])
            if p1 <= p0:
                continue

            mask = (positions >= p0) & (positions <= p1)
            if not np.any(mask):
                continue

            frac = (positions[mask] - p0) / (p1 - p0)
            L_arr[mask] = s0["L"] + frac * (s1["L"] - s0["L"])

            h0, h1 = s0["H"], s1["H"]
            dh = h1 - h0
            if dh > 180:
                dh -= 360
            elif dh < -180:
                dh += 360
            H_arr[mask] = (h0 + frac * dh) % 360

            C_rel_arr[mask] = stop_c_rel[i] + frac * (stop_c_rel[i + 1] - stop_c_rel[i])
            C_abs_arr[mask] = stop_c_abs[i] + frac * (stop_c_abs[i + 1] - stop_c_abs[i])

        last = stops_sorted[-1]
        last_pos = float(last["pos"])
        tail_mask = positions > last_pos
        if np.any(tail_mask):
            c_rel, c_abs = _stop_chroma_rel_abs(last, relative_chroma)
            L_arr[tail_mask] = last["L"]
            H_arr[tail_mask] = last["H"]
            C_rel_arr[tail_mask] = c_rel
            C_abs_arr[tail_mask] = c_abs

    mix = float(np.clip(interp_mix, 0.0, 1.0))
    if mix <= 0.0:
        C_abs = np.clip(C_abs_arr, 0.0, None)
    elif mix >= 1.0:
        C_abs = _chroma_rel_to_abs(L_arr, H_arr, C_rel_arr)
    else:
        C_abs_rel = _chroma_rel_to_abs(L_arr, H_arr, C_rel_arr)
        C_abs = (1.0 - mix) * C_abs_arr + mix * C_abs_rel
        C_abs = np.clip(C_abs, 0.0, None)

    rgb = gamut_map_to_srgb(L_arr, C_abs, H_arr, method="compress")
    return np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)
