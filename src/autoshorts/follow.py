"""Camera path for `--layout follow`: a 9:16 crop window that pans toward the action.

The path is a list of keyframes (time, centre) with the centre as a fraction of the scaled
source width. It starts moving only when the action leaves the middle half of the crop, then
re-centres on it, never faster than `max_speed` widths per second, so it reads as a slow camera
move rather than jitter.
"""

from __future__ import annotations

import math

import numpy as np

from .motion import FloatArray

Keyframes = list[tuple[float, float]]


def camera_path(focus: FloatArray, step: float, window: float, max_speed: float = 0.12) -> Keyframes:
    """Keyframes for the crop centre. `focus[i]` is the action position over [i*step, (i+1)*step)
    (NaN = no opinion), `window` is the crop width as a fraction of the scaled source width."""
    if window >= 1.0 or len(focus) == 0:
        return [(0.0, 0.5)]
    half = window / 2

    def reachable(c: float) -> float:
        return min(1.0 - half, max(half, c))

    targets = smooth(focus)
    first = next((float(v) for v in targets if not math.isnan(v)), 0.5)
    cur = reachable(first)
    keys: Keyframes = [(0.0, cur)]
    moving = False
    limit = max_speed * step
    goal = cur  # last known target, so a still second mid-pan doesn't end the pan
    for i, v in enumerate(targets):
        if not math.isnan(v):
            goal = reachable(float(v))
        target = goal if moving or not math.isnan(v) else cur
        # dead zone with hysteresis: hold still while the action is inside the middle half of the
        # crop, but once a pan starts, keep going until the action is close to the centre again
        # (stopping at the dead zone edge would park every subject a quarter crop off-centre)
        if not moving and abs(target - cur) > window / 4:
            moving = True
        if moving:
            cur += max(-limit, min(limit, target - cur))
            if abs(target - cur) <= window / 16:
                moving = False
        keys.append(((i + 0.5) * step, cur))
    return simplify(keys)


def smooth(focus: FloatArray) -> list[float]:
    """Median of each window and its neighbours (ignoring NaN), so one odd window doesn't start a pan.

    At the ends the 3-window span shifts inward instead of shrinking: the median of 2 values is
    their mean, which would start the clip halfway between two spots."""
    out: list[float] = []
    n = len(focus)
    for i in range(n):
        lo = max(0, min(i - 1, n - 3))
        near = [float(v) for v in focus[lo:lo + 3] if not math.isnan(v)]
        if math.isnan(focus[i]):
            out.append(math.nan)
        elif len(near) < 3:
            # the median of 2 is their mean, a spot where nothing may be. With 2 there's no
            # outlier to reject anyway, so trust this window
            out.append(float(focus[i]))
        else:
            out.append(float(np.median(near)))
    return out


def simplify(keys: Keyframes, eps: float = 1e-4) -> Keyframes:
    """Drops keyframes that sit on the line between their neighbours (holds and steady pans)."""
    if len(keys) <= 2:
        return keys
    out = [keys[0]]
    for k in range(1, len(keys) - 1):
        (t0, c0), (t1, c1), (t2, c2) = out[-1], keys[k], keys[k + 1]
        expected = c0 + (c2 - c0) * (t1 - t0) / (t2 - t0)
        if abs(c1 - expected) > eps:
            out.append(keys[k])
    out.append(keys[-1])
    return out


def centre_expr(keys: Keyframes) -> str:
    """ffmpeg expression in `t` for the piecewise linear centre: c0 plus one clipped ramp per segment.

    No if/else nesting, so it stays flat however many keyframes there are."""
    expr = f"{keys[0][1]:.4f}"
    for (t0, c0), (t1, c1) in zip(keys, keys[1:], strict=False):
        if abs(c1 - c0) < 1e-6:
            continue
        if t1 - t0 < 1e-3:
            # zero-length segment (a jump cut removed the time in between): a hard step.
            # Skipping it would drop its delta and leave every later position off by that much
            expr += f"{c1 - c0:+.4f}*gte(t,{t0:.3f})"
        else:
            expr += f"{c1 - c0:+.4f}*clip((t-{t0:.3f})/{t1 - t0:.3f},0,1)"
    return expr


def crop_x_expr(keys: Keyframes) -> str:
    """x for ffmpeg's crop filter: the centre in pixels minus half the crop, kept inside the frame."""
    return f"clip(({centre_expr(keys)})*iw-ow/2,0,iw-ow)"
