import subprocess
from pathlib import Path


def make_video(path: Path, duration: int = 90) -> None:
    """Synthetic test video: quiet tone, then a loud burst from 40s to 55s, then quiet again."""
    audio = (f"aevalsrc='0.02*sin(2*PI*220*t) + if(between(t,40,55), 0.8*sin(2*PI*660*t), 0)'"
             f":s=44100:d={duration}")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s=640x360:r=30:d={duration}",
         "-f", "lavfi", "-i", audio, "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(path)],
        check=True,
    )
