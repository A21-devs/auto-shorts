# Changelog

## 0.3.0

New:
- `--aspect 9:16|1:1|4:5|16:9`, and `--platform youtube`, `x` and `linkedin` presets with their own default frame (#98)
- `--pick claude|ollama|openai`: an AI rates the transcripts of the best candidate moments and the final pick blends its rating with the signal score (#99)
- `openai` backend for `--pick` and `--copy`: LM Studio by default, or any OpenAI-compatible API via `CLIPWRIGHT_OPENAI_URL` (#99)
- `--caption-style bold|clean|boxed|pop|minimal` (#100)
- `--hook`: a title at the top for the first 3 seconds, from the AI title or the clip's first sentence (#100)
- Give a folder instead of a file to clip every video in it (#101)
- `--copy` also saves `<name>_clipNN.txt` with the post text, ready to paste (#101)
- `--json` for scripts and agents (#101)
- Dockerfile, an agent skill in `skills/clipwright`, and a Windows CI job

Changed:
- `--copy-model` no longer defaults to `llama3.2` for every backend; Ollama still falls back to it

## 0.2.0

- Renamed from auto-shorts to clipwright (the `autoshorts` command still works)
- Speaker layout, folder watcher, MCP server with progress, link downloads, `.srt` and cover per clip, `--copy`, `--jumpcut`, `--platform`, the review page with trim, keep/drop and per-clip layout

## 0.1.0

- First version: signal-based clip picking, blur/crop/follow/fit layouts, word-by-word captions, loudness normalisation
