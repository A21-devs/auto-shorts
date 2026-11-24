"""Title, description and hashtags for each clip, written by a local AI (optional).

Two backends, both without an API key:

- claude: the Claude Code CLI in print mode (`claude -p`), using whatever account it's signed
  in with. The transcript goes in on stdin, a JSON schema makes it answer in a fixed shape.
- ollama: a model running in a local Ollama server (http://127.0.0.1:11434), with the same
  schema passed as `format`.

Only the clip's transcript text is sent, never audio or video. Nothing runs unless asked for
with --copy. If a call fails the clip is still fine, it just has no copy.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .transcribe import Word

BACKENDS = ("claude", "ollama")
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "llama3.2"
TIMEOUT = 120  # seconds per clip
MAX_WORDS = 600  # a 60s clip is ~150-200 words, this only guards against huge ranges

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "under 70 characters, no hashtags"},
        "description": {"type": "string", "description": "one or two sentences"},
        "hashtags": {"type": "array", "items": {"type": "string"}, "description": "3 to 6, without #"},
    },
    "required": ["title", "description", "hashtags"],
    "additionalProperties": False,
}

SYSTEM = (
    "You write the post text for a short vertical video cut from a longer one. You get what is said "
    "in the clip. Write a title that makes people want to watch (no clickbait lies, no emoji, under 70 "
    "characters), a one or two sentence description, and 3 to 6 relevant hashtags without the # sign. "
    "Use the language of the transcript. Only describe what is actually in the transcript."
)


class CopyError(RuntimeError):
    pass


@dataclass(frozen=True)
class PostCopy:
    title: str
    description: str
    hashtags: list[str]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def prompt_for(words: Sequence[Word], platform: str | None) -> str:
    text = " ".join(w.text for w in words[:MAX_WORDS])
    where = f" It will be posted on {platform}." if platform else ""
    return f"Transcript of the clip:{where}\n\n{text}"


def parse(data: Any) -> PostCopy:
    """Checks and tidies what a model returned."""
    if not isinstance(data, dict):
        raise CopyError("the answer wasn't a JSON object")
    title, desc, tags = data.get("title"), data.get("description"), data.get("hashtags")
    if not isinstance(title, str) or not title.strip() or not isinstance(desc, str) or not isinstance(tags, list):
        raise CopyError("the answer was missing the title, description or hashtags")
    clean: list[str] = []
    for tag in tags:
        t = "".join(str(tag).split()).lstrip("#")
        if t and t.lower() not in {c.lower() for c in clean}:
            clean.append(t)
    return PostCopy(title.strip()[:100], desc.strip(), clean[:6])


def ask_claude(prompt: str, command: Sequence[str] = ("claude",)) -> PostCopy:
    exe = shutil.which(command[0])
    if exe is None:
        raise CopyError("the claude command isn't installed (https://claude.com/claude-code)")
    args = [exe, *command[1:], "-p", "--output-format", "json", "--tools", "", "--no-session-persistence",
            "--system-prompt", SYSTEM, "--json-schema", json.dumps(SCHEMA)]
    try:
        proc = subprocess.run(args, input=prompt, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        raise CopyError(f"claude didn't answer within {TIMEOUT}s") from None
    try:
        out = json.loads(proc.stdout)
    except ValueError:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["no output"]
        raise CopyError(f"claude failed: {tail[0]}") from None
    if not isinstance(out, dict):
        raise CopyError("claude returned something unexpected")
    if out.get("is_error"):
        text = str(out.get("result") or "unknown error")
        if any(k in text.lower() for k in ("auth", "log in", "logged in")):
            raise CopyError("Claude Code isn't signed in, run `claude auth login` once")
        raise CopyError(f"claude failed: {text.splitlines()[0]}")
    return parse(out.get("structured_output"))


def ask_ollama(prompt: str, model: str = DEFAULT_OLLAMA_MODEL, url: str = OLLAMA_URL) -> PostCopy:
    body = json.dumps({"model": model, "system": SYSTEM, "prompt": prompt, "format": SCHEMA, "stream": False,
                       "options": {"temperature": 0.7}}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as res:
            out = json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        if e.code == 404 and "not found" in detail:
            raise CopyError(f"Ollama doesn't have {model}, run `ollama pull {model}` once") from None
        raise CopyError(f"Ollama answered {e.code}: {detail[:200]}") from None
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise CopyError(f"can't reach Ollama at {url} ({e}), is it running?") from None
    try:
        return parse(json.loads(out["response"]))
    except (KeyError, TypeError, ValueError):
        raise CopyError("Ollama's answer wasn't the JSON that was asked for") from None


def write_copy(words: Sequence[Word], backend: str, platform: str | None = None,
               ollama_model: str = DEFAULT_OLLAMA_MODEL) -> PostCopy:
    if not words:
        raise CopyError("no transcript for this clip")
    prompt = prompt_for(words, platform)
    if backend == "claude":
        return ask_claude(prompt)
    if backend == "ollama":
        return ask_ollama(prompt, ollama_model)
    raise ValueError(f"unknown backend {backend!r}")
