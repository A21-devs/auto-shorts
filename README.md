# clipwright

Turn a long video into short clips ready to post: the best moments, cut, reframed, captioned, with a cover and the post text. Runs on your own machine, no account, no API key needed.

[![CI](https://img.shields.io/github/actions/workflow/status/A21-devs/clipwright/ci.yml?branch=main&label=CI)](https://github.com/A21-devs/clipwright/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/A21-devs/clipwright?label=release)](https://github.com/A21-devs/clipwright/releases/latest)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB)](#install)
[![License: MIT](https://img.shields.io/github/license/A21-devs/clipwright)](LICENSE)
[![Stars](https://img.shields.io/github/stars/A21-devs/clipwright?style=flat)](https://github.com/A21-devs/clipwright/stargazers)

Inspired by [AutoClip](https://github.com/zhouxiaoka/autoclip) by @zhouxiaoka. Same idea, written from scratch: a small Python tool you run from the terminal, a local web page, a folder watcher, or an AI agent.

```bash
pip install "clipwright[transcribe,link] @ git+https://github.com/A21-devs/clipwright"
clipwright "my 2 hour stream.mp4" -n 8
clipwright "https://www.youtube.com/watch?v=..." --platform tiktok --pick claude --copy claude --hook
```

**What you get for each clip:** a 9:16 (or square, 4:5, 16:9) MP4 with word-by-word burned-in captions, audio at -14 LUFS, a cover image, an `.srt` for upload, and optionally an AI-written title, description and hashtags in a text file ready to paste.

## Features

- **Finds the moments**: loudness, peaks, the first 3 seconds, talking speed and scene cuts, snapped to pauses and sentence ends. Optionally an AI reads the transcripts of the best candidates and helps choose (`--pick`).
- **Any source**: a file, a folder of files, a YouTube / Twitch / Kick link, or an OBS recordings folder it watches.
- **Reframes**: blurred background, center crop, a camera that follows the action (gameplay), or one that cuts to whoever is talking (podcasts).
- **Captions**: word-by-word with the current word highlighted, five styles, a hook title for the first 3 seconds, and pauses cut out with captions retimed (`--jumpcut`).
- **Platforms**: Shorts, TikTok, Reels, YouTube, X and LinkedIn presets with the right lengths and frame.
- **Post text**: title, description and hashtags from Claude Code, Ollama, LM Studio or any OpenAI-compatible API.
- **Review page**: watch every clip, trim it, change its layout, keep or drop it.
- **For agents**: an MCP server, an agent skill, and `--json` output.
- **Private**: video never leaves your machine. The optional AI features send transcript text only.

### clipwright and AutoClip

AutoClip is the bigger project, and if you want a desktop app it's the one to use. clipwright is the terminal-first take on the same idea.

| | clipwright | AutoClip |
| --- | --- | --- |
| How you run it | CLI, local web page, folder watcher, MCP, Docker | Desktop app (Windows, macOS), web UI in Docker, CLI, MCP |
| Picks moments with | audio + scene signals, optionally blended with an AI rating of the transcript | an AI model reading the transcript, vision models for gameplay |
| AI providers | Claude Code (no key), Ollama, LM Studio / any OpenAI-compatible API | many cloud providers, Ollama, LM Studio |
| Reframing | blur, crop, follow the action, cut to the speaker | speaker tracking, smart reframing |
| Captions | 5 styles, hook title, jump cuts | captions, covers with titles, 13 UI languages |
| Publishing | writes the post text, you upload | can publish directly to some platforms |
| Size | one Python package, numpy + ffmpeg (extras optional) | desktop app with bundled Python and FFmpeg |

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

`--platform shorts|tiktok|reels|youtube|x|linkedin` picks clip lengths that fit: Shorts 15-58s (hard limit 60), TikTok 15-60s (limit 3 min), Reels 15-60s (limit 90s), YouTube 1-5 min landscape, X 15-120s landscape (limit 2:20), LinkedIn 15-90s square. `--min`/`--max` still win, but a `--max` over the platform's limit is an error instead of an upload that gets rejected later. The platform is saved in `manifest.json`.

`--aspect 9:16|1:1|4:5|16:9` picks the frame. Vertical 9:16 is the default; `youtube` and `x` default to 16:9 and `linkedin` to 1:1, and `--aspect` overrides either. Every layout, the captions and the covers follow the frame, so a landscape clip of a podcast still gets the speaker layout and readable captions.

## Output

For each clip:
- 1080x1920 H.264 MP4. It uses NVENC if you have an NVIDIA GPU, otherwise x264.
- Audio normalized to -14 LUFS (the usual target for shorts platforms).
- Captions: word-by-word with the current word highlighted, burned in. The `.ass` file is kept next to the video if you want to edit it.
- A cover `.jpg` per clip (720x1280), taken from the finished clip near its loudest moment, picking the sharpest frame within a second of it.
- A plain `.srt` per clip (normal subtitle lines, up to 42 characters) to upload next to the video on platforms that take one.
- `manifest.json` with every clip's time range, score and why it was picked.

Layouts (`--layout`):
- `blur` (default): the full video fits the width, with a blurred zoomed copy behind it
- `crop`: fills the frame (center crop)
- `follow`: fills the frame and pans slowly toward where the motion is (good for gameplay)
- `speaker`: fills the frame and cuts to whoever is talking, for podcasts and interviews. Finds faces with OpenCV and watches whose mouth moves, so it needs `pip install "clipwright[speaker]"`
- `fit`: black bars

Caption styles (`--caption-style`):
- `bold` (default): all caps, white with the current word in yellow
- `clean`: mixed case, green highlight, the podcast-clip look
- `boxed`: on a dark box, for busy gameplay where an outline gets lost
- `pop`: huge, two words at a time, pink highlight
- `minimal`: small and low, no highlight

`--hook` puts a title at the top for the first 3 seconds, the part that decides whether someone keeps watching. It's the `--copy` title when there is one, otherwise the clip's first sentence. Re-rendering a clip on the review page keeps it.

`--jumpcut` cuts pauses longer than half a second out of each clip, keeping 0.1s on each side so words aren't clipped. Captions are retimed to match. A pause is anything 25 dB under the clip's loud parts, so constant game audio is left alone.

## Links

```bash
pip install "clipwright[link] @ git+https://github.com/A21-devs/clipwright"
clipwright "https://www.youtube.com/watch?v=..." -n 6 --platform shorts
```

Anything [yt-dlp](https://github.com/yt-dlp/yt-dlp) can download works (YouTube, Twitch VODs, Kick, ...). It's saved as `<title> [<id>].mp4` in the current folder (or `--downloads DIR`), capped at 1080p, and pasting the same link again reuses the file. The video's subtitles come along as `<title> [<id>].srt` when there are any, so Whisper is skipped.

## Already have subtitles?

If there's a `talk.srt` next to `talk.mp4` it's used instead of Whisper, which skips the slowest step. Or point at one with `--srt path/to/file.srt`. SRT only has timings per line, so each line's time is spread over its words by length for the word highlight.

## Let an AI help pick the clips (optional)

```bash
clipwright podcast.mp4 -n 6 --pick claude                  # Claude Code CLI, no API key
clipwright podcast.mp4 --pick ollama --copy-model qwen3     # a local Ollama model
clipwright podcast.mp4 --pick openai --copy-model qwen3-8b  # LM Studio, or any OpenAI-compatible server
```

Loudness and scene cuts are good at finding where something happens, not at telling whether what's said works on its own. With `--pick`, clipwright takes a shortlist of the best moments by signal (3x as many as you asked for), sends their transcripts to the model, and asks it to rate each one as a standalone short: does it hook, does it make sense without the rest, does it pay off. The final pick blends the two, and each clip's reasons in `manifest.json` and on the review page say what the model thought ("AI 8/10: clear story with a punchline"). Only transcript text is sent. If the model can't be reached the run goes on with the normal pick.

`openai` talks to LM Studio's local server (`http://127.0.0.1:1234/v1`) by default. Point it anywhere else that speaks the OpenAI chat API (OpenAI, OpenRouter, vLLM, llama.cpp) with `CLIPWRIGHT_OPENAI_URL`, and set `OPENAI_API_KEY` if that server needs one. The model is `--copy-model` or `CLIPWRIGHT_OPENAI_MODEL`. `--copy` takes the same three backends.

## Titles and hashtags (optional AI)

```bash
clipwright stream.mp4 --platform tiktok --copy claude     # Claude Code CLI, no API key
clipwright stream.mp4 --copy ollama --copy-model qwen3    # a local Ollama model
clipwright stream.mp4 --copy openai --copy-model gpt-5-mini  # OpenAI-compatible (see above)
```

`--copy` writes a title, a short description and 3-6 hashtags for each clip from what's said in it. They go in `manifest.json`, show on the clip cards in `serve`, and are saved as `<name>_clipNN.txt` next to the clip, ready to paste when you post. Only the clip's transcript text is sent, never audio or video, and nothing is sent unless you pass `--copy`. `claude` uses the account the Claude Code CLI is signed in with (`claude auth login`). If a call fails, the clip is still rendered, just without the text.

## A whole folder at once

```bash
clipwright "D:/Videos/OBS" -n 5 --platform tiktok
```

Give it a folder instead of a file and it clips every video directly inside it, one after the other. Videos that already have clips are skipped, so you can run it again after adding more, and one video failing doesn't stop the rest. Each video uses its own `<name>.srt` if there is one.

## For scripts and agents: --json

`--json` prints the finished `manifest.json` on stdout (for a folder: which videos were done, skipped or failed) and moves the progress log to stderr, so `clipwright talk.mp4 --json | jq .clips` works.

## Watch a recordings folder

```
clipwright watch "D:/Videos/OBS" --platform shorts --layout follow
```

Waits for new recordings in the folder and clips each one once it's finished. A file counts as finished when it has stopped growing for 10 seconds (`--settle`), so it works with OBS without any plugin. Partial downloads (`.part`, `.tmp`) are ignored, subfolders aren't watched, and recordings that already have a `manifest.json` are skipped, so you can stop and restart it any time. Takes the same clip options as a normal run; `-o` sets where the `<name>_shorts` folders go.

## Review in the browser

```bash
clipwright serve
```

Open http://127.0.0.1:8765/, paste the path of a video (or a link, with `clipwright[link]` installed) and hit Make shorts. Links are downloaded to `Videos/clipwright` unless you start it with `--downloads DIR`. It shows progress while it works, then every clip with its score and why it was picked. The server only listens on your own machine.

Before 0.2 this was called **auto-shorts**. The `autoshorts` command still works, and analysis cached under the old name is picked up.

## Use it from an AI agent (MCP)

```
pip install "clipwright[mcp]"
claude mcp add clipwright -- clipwright mcp
```

`clipwright mcp` is an MCP server over stdio, so Claude Code, Claude Desktop and other MCP clients can run it. Tools:
- `make_clips(video, count, platform, layout, captions, jumpcut, model, out, downloads, aspect, pick, caption_style, hook)` runs a job on a local file or a link and returns the clips, with progress notifications while it works
- `list_clips(manifest)` reads a finished run
- `rerender_clip(manifest, n, start, end, layout)` redoes one clip with a new range or layout

Same limits as the review page.

### Agent skill

[`skills/clipwright/SKILL.md`](skills/clipwright/SKILL.md) teaches a coding agent (Claude Code and others that read skills) when and how to run clipwright from the shell: which options fit which kind of video, and how to read the manifest. Copy the folder into `~/.claude/skills/` (or your project's `.claude/skills/`).

## Install

Needs Python 3.12+ and [ffmpeg](https://ffmpeg.org/) on your PATH.

```bash
pip install "clipwright[transcribe] @ git+https://github.com/A21-devs/clipwright"
# NVIDIA GPU? this pulls in the CUDA libraries so Whisper runs on the GPU:
pip install "clipwright[gpu] @ git+https://github.com/A21-devs/clipwright"
```

Or download the wheel from the [latest release](https://github.com/A21-devs/clipwright/releases/latest) and `pip install "clipwright-<version>-py3-none-any.whl[transcribe]"`.

Extras, add them in the brackets as needed: `transcribe` (Whisper captions), `gpu` (Whisper on an NVIDIA GPU), `link` (YouTube/Twitch links), `speaker` (the speaker layout), `mcp` (the MCP server). Without `transcribe` it still works; it picks clips from audio + scenes and skips captions.

### Docker

```bash
docker build -t clipwright https://github.com/A21-devs/clipwright.git
docker run --rm -v "$PWD:/work" clipwright "/work/my stream.mp4" -n 6 --platform shorts
```

The image has ffmpeg and every extra except `gpu`. Clips land next to the video in the mounted folder. Whisper models are downloaded on first use; mount a cache folder at `/root/.cache` to keep them between runs.

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
