---
name: clipwright
description: Cut a long video (stream VOD, podcast, gameplay, a YouTube/Twitch link) into short captioned clips for Shorts, TikTok, Reels, YouTube, X or LinkedIn with the clipwright CLI. Use when someone asks to make shorts, clips or highlights from a video file, a folder of recordings, or a link.
---

# Making clips with clipwright

clipwright runs locally: it picks the best moments, reframes them, burns in word-by-word captions, and writes a cover, an `.srt` and a `manifest.json` per run. Check it's installed with `clipwright --version`; ffmpeg must be on PATH.

## Pick the options from the video

| The video is | Use |
| --- | --- |
| a podcast or interview with people on camera | `--layout speaker` (needs the `speaker` extra) |
| gameplay or anything where the action moves | `--layout follow` |
| a talk, a screen recording, or unknown | the default `--layout blur` |
| for a specific platform | `--platform shorts\|tiktok\|reels\|youtube\|x\|linkedin` (sets lengths and frame) |
| long with lots of dead air | add `--jumpcut` |

Add `--pick claude` when what's said matters more than how loud it is (podcasts, talks), and `--copy claude --hook` when the person wants titles, hashtags and an on-screen hook. These send transcript text only, to the Claude Code CLI on this machine. Use `ollama` or `openai` (LM Studio, or `CLIPWRIGHT_OPENAI_URL`) instead of `claude` if that's what they have.

## Run it

```bash
clipwright "path/to/video.mp4" -n 6 --platform shorts --json > run.json
clipwright "https://www.youtube.com/watch?v=..." -n 6 --json      # needs the link extra
clipwright "path/to/folder" -n 4 --json                             # every video in the folder
```

`--json` prints the manifest on stdout and the progress log on stderr. A 1 hour video takes a few minutes with a GPU, longer on CPU (Whisper is the slow part; a `<name>.srt` next to the video skips it). Run it in the background for long videos.

## Read the result

Clips go to `<video name>_shorts/` next to the video (or `-o DIR`). In the manifest, each entry of `clips` has `file`, `start`/`end` in the source, `length`, `score`, `reasons` (why it was picked, including `AI 8/10: ...` with `--pick`), `thumbnail`, `srt`, and with `--copy` a `copy` object plus `post` (a `.txt` ready to paste).

Tell the person which clips came out, how long each is and why it was picked, and where the folder is. To change one clip's range or layout, run the MCP tool `rerender_clip` or open `clipwright serve` and use the review page.

## Don't

- Don't upload or post the clips anywhere. clipwright stops at files on disk.
- Don't pass `--pick` or `--copy` with `--no-captions`; both read the transcript.
