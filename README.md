# clipwright

Inspired by [AutoClip](https://github.com/zhouxiaoka/autoclip) by @zhouxiaoka. Same idea, written from scratch: a small Python tool you run from the terminal or a local web page.

Give it a long video (a stream VOD, a gameplay recording, a podcast) and it finds the best moments and cuts them into vertical 9:16 shorts with word-by-word captions. It runs on your own machine and doesn't need an API key.

```bash
clipwright "my 2 hour stream.mp4" -n 8
```

```
my 2 hour stream.mp4: 124.3 min, 1920x1080 @ 60fps
reading audio
finding scene cuts
transcribing (whisper small)
analysis done in 71.4s (412 cuts, 18233 words)
[1/8]   412.5s -   455.0s  score  3.41  loud moment, strong start, lots of talking
         rendered my 2 hour stream_clip01.mp4 in 6.2s
...
```

## How it picks clips

Everything is on a 0.5 second timeline:

| signal | where it comes from | why |
| --- | --- | --- |
| loudness | RMS of the audio, z-scored against the whole video | yelling, explosions and laughing are usually the good parts |
| peaks | 95th percentile loudness inside a window | something actually happens |
| hook | loudness in the first 3 seconds | shorts live or die on the start |
| speech | words per second from Whisper | talking segments make better shorts than dead air |
| scene cuts | ffmpeg's `scene` score on a tiny 5fps copy | some cuts = action, tons of cuts = loading screens (penalized) |

Every window between `--min` and `--max` seconds gets a score. Start and end points get snapped to silences or sentence ends so clips don't cut mid-word. Then the best non-overlapping windows are picked.

## Platforms

`--platform shorts|tiktok|reels` picks clip lengths that fit: Shorts 15-58s (hard limit 60), TikTok 15-60s (limit 3 min), Reels 15-60s (limit 90s). `--min`/`--max` still win, but a `--max` over the platform's limit is an error instead of an upload that gets rejected later. The platform is saved in `manifest.json`.

## Output

For each clip:
- 1080x1920 H.264 MP4. It uses NVENC if you have an NVIDIA GPU, otherwise x264.
- Audio normalized to -14 LUFS (the usual target for shorts platforms).
- Captions: word-by-word with the current word highlighted, burned in. The `.ass` file is kept next to the video if you want to edit it.
- `manifest.json` with every clip's time range, score and why it was picked.

Layouts (`--layout`):
- `blur` (default): the full video fits the width, with a blurred zoomed copy behind it
- `crop`: fills the frame (center crop)
- `follow`: fills the frame and pans slowly toward where the motion is (good for gameplay)
- `fit`: black bars

`--jumpcut` cuts pauses longer than half a second out of each clip, keeping 0.1s on each side so words aren't clipped. Captions are retimed to match. A pause is anything 25 dB under the clip's loud parts, so constant game audio is left alone.

## Links

```bash
pip install "clipwright[link] @ git+https://github.com/A21-devs/clipwright"
clipwright "https://www.youtube.com/watch?v=..." -n 6 --platform shorts
```

Anything [yt-dlp](https://github.com/yt-dlp/yt-dlp) can download works (YouTube, Twitch VODs, Kick, ...). It's saved as `<title> [<id>].mp4` in the current folder (or `--downloads DIR`), capped at 1080p, and pasting the same link again reuses the file. The video's subtitles come along as `<title> [<id>].srt` when there are any, so Whisper is skipped.

## Already have subtitles?

If there's a `talk.srt` next to `talk.mp4` it's used instead of Whisper, which skips the slowest step. Or point at one with `--srt path/to/file.srt`. SRT only has timings per line, so each line's time is spread over its words by length for the word highlight.

## Review in the browser

```bash
clipwright serve
```

Open http://127.0.0.1:8765/, paste the path of a video and hit Make shorts. It shows progress while it works, then every clip with its score and why it was picked. The server only listens on your own machine.

Before 0.2 this was called **clipwright**. The `clipwright` command still works, and analysis cached by the old name is picked up.

Before 0.2 this was called **auto-shorts**. The `autoshorts` command still works, and analysis cached under the old name is picked up.

## Install

Needs Python 3.12+ and [ffmpeg](https://ffmpeg.org/) on your PATH.

```bash
pip install "clipwright[transcribe] @ git+https://github.com/A21-devs/clipwright"
# NVIDIA GPU? this pulls in the CUDA libraries so Whisper runs on the GPU:
pip install "clipwright[gpu] @ git+https://github.com/A21-devs/clipwright"
```

Without the `transcribe` extra it still works. It just picks clips from audio + scenes and skips captions.

## Development

```bash
uv sync --all-extras
uv run pytest        # includes an end-to-end test that renders a synthetic video
uv run ruff check src tests
uv run mypy
```

## Credits

Built by me, with help from [Claude](https://claude.ai) (Anthropic) as a coding assistant.

## License

MIT
