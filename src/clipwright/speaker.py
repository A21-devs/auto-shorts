"""Faces for `--layout speaker`: where the faces are and whose mouth is moving.

Decodes a grayscale copy of the clip (5 fps, 480 px wide) through an ffmpeg pipe and runs a face
detector on every frame. For each face it also measures mouth activity: how much the lower middle
of the face box changed since the previous frame. On a podcast the person talking has a moving
mouth and the listener mostly doesn't, which is a better hint than face size.

The detector is OpenCV's frontal face Haar cascade. It ships inside the opencv wheel, so there's
no model file to download (YuNet is more accurate but needs one). OpenCV is an optional extra:
`pip install "clipwright[speaker]"`.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .follow import Keyframes
from .motion import small_size
from .probe import require

Gray = npt.NDArray[np.uint8]
Box = tuple[int, int, int, int]  # x, y, w, h in pixels of the analysed frame
Detector = Callable[[Gray], Sequence[Box]]


@dataclass(frozen=True)
class Face:
    """One face in one frame. Position and size are fractions of the frame (0-1)."""
    x: float
    y: float
    w: float
    h: float
    mouth: float  # mean abs change of the mouth area since the previous frame, 0-255

    @property
    def centre(self) -> float:
        return self.x + self.w / 2

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass(frozen=True)
class Faces:
    fps: float
    frames: list[list[Face]]  # frame i is at t = i / fps

    @property
    def duration(self) -> float:
        return len(self.frames) / self.fps


def available() -> bool:
    try:
        import cv2  # noqa: F401
    except ImportError:
        return False
    return True


def haar_detector(min_face: float = 0.08) -> Detector:
    """OpenCV's frontal face cascade. Faces smaller than `min_face` of the frame height are
    ignored: on a podcast those are posters and people in the background."""
    try:
        import cv2
    except ImportError:
        raise RuntimeError('the speaker layout needs OpenCV: pip install "clipwright[speaker]"') from None

    # cv2.data is a plain package next to the extension module, the bundled stubs leave it out
    cascade_dir: str = cv2.data.haarcascades  # type: ignore[attr-defined]
    cascade = cv2.CascadeClassifier(cascade_dir + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        raise RuntimeError("OpenCV's face cascade is missing from this install")

    def detect(gray: Gray) -> Sequence[Box]:
        side = max(16, int(gray.shape[0] * min_face))
        found = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(side, side))
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in found]

    return detect


def mouth_box(box: Box) -> Box:
    """The lower middle of a face box, where the mouth and jaw are."""
    x, y, w, h = box
    return x + w // 4, y + (h * 2) // 3, max(1, w // 2), max(1, h // 3)


def eye_box(box: Box) -> Box:
    """A band across the eyes and nose, the same width as the mouth box. It moves with the head
    but not with the jaw."""
    x, y, w, h = box
    return x + w // 4, y + h // 4, max(1, w // 2), max(1, h // 4)


def _change(cur: Gray, prev: Gray, box: Box) -> float:
    x, y, w, h = box
    a = cur[y:y + h, x:x + w].astype(np.int16)
    b = prev[y:y + h, x:x + w].astype(np.int16)
    return float(np.abs(a - b).mean()) if a.size else 0.0


def mouth_change(cur: Gray, prev: Gray | None, box: Box) -> float:
    """How much the mouth moved relative to the rest of the face. The eye band's change is
    taken off, so a head that nods or shifts doesn't count as talking (a 3 px head shift
    scored 23.8 without this, more than real mouth movement)."""
    if prev is None:
        return 0.0
    return max(0.0, _change(cur, prev, mouth_box(box)) - _change(cur, prev, eye_box(box)))


def faces_in(cur: Gray, prev: Gray | None, detect: Detector) -> list[Face]:
    h, w = cur.shape
    out = []
    for box in detect(cur):
        bx, by, bw, bh = box
        out.append(Face(bx / w, by / h, bw / w, bh / h, mouth_change(cur, prev, box)))
    return out


def read_faces(path: Path, video_stream: int, width: int, height: int, start: float | None = None,
               duration: float | None = None, fps: float = 5.0, size: int = 480,
               detect: Detector | None = None) -> Faces:
    """Faces in [start, start + duration) (the whole video by default), sampled at `fps`."""
    detect = detect or haar_detector()
    w, h = small_size(width, height, size)
    cmd = [require("ffmpeg"), "-v", "error", "-nostdin"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(path), "-map", f"0:{video_stream}", "-an",
            "-vf", f"fps={fps:g},scale={w}:{h}:flags=area,format=gray", "-f", "rawvideo", "-"]
    frame_bytes = w * h
    frames: list[list[Face]] = []
    # same pipe handling as motion.read_motion: stderr to a file so a chatty ffmpeg can't
    # deadlock us, and kill ffmpeg if we stop reading early
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        assert proc.stdout is not None
        prev: Gray | None = None
        finished = False
        try:
            while True:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                cur = np.frombuffer(buf, dtype=np.uint8).reshape(h, w)
                frames.append(faces_in(cur, prev, detect))
                prev = cur
            finished = True
        finally:
            proc.stdout.close()
            if not finished:
                proc.kill()
            code = proc.wait()
        if code != 0:
            err.seek(0)
            tail = "\n".join(err.read().decode(errors="replace").strip().splitlines()[-5:])
            raise RuntimeError(f"face analysis failed for {path.name}:\n{tail}")
    return Faces(fps, frames)


# --- who to show -------------------------------------------------------------------------------

MIN_PRESENCE = 0.3  # a face position seen in fewer frames than this is a false hit or a passer-by...
MIN_SEEN = 2.0      # ...unless it's there for this many seconds (a short wide shot in an edited podcast)
TRACK_GAP = 0.08    # face centres further apart than this (fraction of width) are different people
TALKING = 2.0       # mean mouth activity above which someone counts as talking
LEAD = 1.5          # ...and they have to beat the next person by this factor


@dataclass(frozen=True)
class Track:
    """One person: a stretch of the frame where a face keeps showing up."""
    lo: float      # leftmost and rightmost face centre seen
    hi: float
    centre: float  # median centre
    area: float    # median face area
    presence: float  # share of frames with this face in them


def tracks(faces: Faces, min_presence: float = MIN_PRESENCE, gap: float = TRACK_GAP,
           min_seen: float = MIN_SEEN) -> list[Track]:
    """Groups face centres into people, left to right. Podcast cameras are static and people stay
    in their seat, so 1-D clustering on the centre is enough: sort, split at gaps.

    A position counts as a person when it's in `min_presence` of the frames or seen for
    `min_seen` seconds. The relative bar alone dropped both people from a 4 s wide shot followed
    by 14 s of close-ups; the absolute one still drops the single-frame false hits on gameplay."""
    seen = sorted((f.centre, f.area, i) for i, frame in enumerate(faces.frames) for f in frame)
    if not seen:
        return []
    groups: list[list[tuple[float, float, int]]] = [[seen[0]]]
    for item in seen[1:]:
        if item[0] - groups[-1][-1][0] > gap:
            groups.append([])
        groups[-1].append(item)
    out = []
    for g in groups:
        frames_seen = len({i for _, _, i in g})
        presence = frames_seen / len(faces.frames)
        if presence >= min_presence or frames_seen >= min_seen * faces.fps:
            out.append(Track(g[0][0], g[-1][0], float(np.median([c for c, _, _ in g])),
                             float(np.median([a for _, a, _ in g])), presence))
    return out


def track_of(face: Face, people: list[Track], gap: float = TRACK_GAP) -> int | None:
    for k, t in enumerate(people):
        if t.lo - gap / 2 <= face.centre <= t.hi + gap / 2:
            return k
    return None


def talking(faces: Faces, people: list[Track], step: float = 1.0) -> list[list[float]]:
    """Mean mouth activity of each person per `step` seconds (0 where they aren't seen)."""
    per = max(1, round(step * faces.fps))
    out = []
    for lo in range(0, len(faces.frames), per):
        sums = [0.0] * len(people)
        counts = [0] * len(people)
        # frame `lo` has no previous frame to compare with in the first window, skip it there
        for frame in faces.frames[max(1, lo):lo + per]:
            for f in frame:
                k = track_of(f, people)
                if k is not None:
                    sums[k] += f.mouth
                    counts[k] += 1
        out.append([s / c if c else 0.0 for s, c in zip(sums, counts, strict=True)])
    return out


def speaker_path(faces: Faces, window: float, step: float = 1.0, min_shot: float = 2.0,
                 confirm: int = 2) -> Keyframes:
    """Keyframes for the crop centre that sit on whoever is talking and cut (not pan) between
    people. `window` is the crop width as a fraction of the scaled source width.

    Starts on whoever talks first (the biggest face if nobody clearly does) and switches when
    someone else clearly talks for `confirm` windows in a row, cutting at the start of that run.
    A one-second "yeah" from the listener doesn't cut away and back, and there's never a second
    cut within `min_shot` seconds of the last one."""
    if window >= 1.0:
        return [(0.0, 0.5)]
    people = tracks(faces)
    if not people:
        return [(0.0, 0.5)]
    half = window / 2

    def centre(k: int) -> float:
        return min(1.0 - half, max(half, people[k].centre))

    scores = talking(faces, people, step)
    clear = [clear_speaker(s) for s in scores]
    first = next((k for k in clear if k is not None), max(range(len(people)), key=lambda k: people[k].area))
    cur, last_cut = first, 0.0
    keys: Keyframes = [(0.0, centre(cur))]
    for i, k in enumerate(clear):
        t = i * step
        if k is None or k == cur or t - last_cut < min_shot:
            continue
        run = clear[i:i + confirm]
        if len(run) < confirm or any(c != k for c in run):
            continue  # not long enough (a run cut short by the end of the clip doesn't count either)
        keys += [(t, centre(cur)), (t, centre(k))]
        cur, last_cut = k, t
    end = len(faces.frames) / faces.fps
    if end > keys[-1][0]:
        keys.append((end, centre(cur)))
    return keys


def clear_speaker(scores: list[float]) -> int | None:
    """The one person clearly talking in a window, or None when nobody is or it's a tie."""
    if not scores:
        return None
    ranked = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    best = scores[ranked[0]]
    second = scores[ranked[1]] if len(ranked) > 1 else 0.0
    if best < TALKING or best < LEAD * second:
        return None
    return ranked[0]
