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
