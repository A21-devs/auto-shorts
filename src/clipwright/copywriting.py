"""Title, description and hashtags for each clip, written by a local AI (optional).

Three backends:

- claude: the Claude Code CLI in print mode (`claude -p`), using whatever account it's signed
  in with. The transcript goes in on stdin, a JSON schema makes it answer in a fixed shape.
- ollama: a model running in a local Ollama server (http://127.0.0.1:11434), with the same
  schema passed as `format`.
- openai: any OpenAI-compatible chat completions server. LM Studio on this machine by default,
  or OpenAI, OpenRouter, vLLM, llama.cpp with CLIPWRIGHT_OPENAI_URL (and OPENAI_API_KEY).

Only the clip's transcript text is sent, never audio or video. Nothing runs unless asked for
with --copy. If a call fails the clip is still fine, it just has no copy.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .transcribe import Word

BACKENDS = ("claude", "ollama", "openai")
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OPENAI_URL = "http://127.0.0.1:1234/v1"  # LM Studio's local server
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
    """setup=True: a problem with the backend itself (not installed, not signed in, not running),
    which will be the same for every clip, so there's no point trying the next one."""

    def __init__(self, message: str, setup: bool = False) -> None:
        super().__init__(message)
        self.setup = setup


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


def claude_json(prompt: str, system: str, schema: dict[str, Any], command: Sequence[str] = ("claude",)) -> Any:
    """What the Claude Code CLI answers, in the shape `schema` asks for (not yet checked)."""
    exe = shutil.which(command[0])
    if exe is None:
        raise CopyError("the claude command isn't installed (https://claude.com/claude-code)", setup=True)
    args = [exe, *command[1:], "-p", "--output-format", "json", "--tools", "", "--no-session-persistence",
            "--system-prompt", system, "--json-schema", json.dumps(schema)]
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
            raise CopyError("Claude Code isn't signed in, run `claude auth login` once", setup=True)
        raise CopyError(f"claude failed: {text.splitlines()[0]}")
    return out.get("structured_output")


def ask_claude(prompt: str, command: Sequence[str] = ("claude",)) -> PostCopy:
    return parse(claude_json(prompt, SYSTEM, SCHEMA, command))


def ollama_json(prompt: str, system: str, schema: dict[str, Any], model: str = DEFAULT_OLLAMA_MODEL,
                url: str = OLLAMA_URL) -> Any:
    body = json.dumps({"model": model, "system": system, "prompt": prompt, "format": schema, "stream": False,
                       "options": {"temperature": 0.7}}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as res:
            out = json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        if e.code == 404 and "not found" in detail:
            raise CopyError(f"Ollama doesn't have {model}, run `ollama pull {model}` once", setup=True) from None
        raise CopyError(f"Ollama answered {e.code}: {detail[:200]}") from None
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise CopyError(f"can't reach Ollama at {url} ({e}), is it running?", setup=True) from None
    try:
        return json.loads(out["response"])
    except (KeyError, TypeError, ValueError):
        raise CopyError("Ollama's answer wasn't the JSON that was asked for") from None


def ask_ollama(prompt: str, model: str = DEFAULT_OLLAMA_MODEL, url: str = OLLAMA_URL) -> PostCopy:
    return parse(ollama_json(prompt, SYSTEM, SCHEMA, model, url))


def openai_url() -> str:
    return os.environ.get("CLIPWRIGHT_OPENAI_URL", DEFAULT_OPENAI_URL).rstrip("/")


def openai_json(prompt: str, system: str, schema: dict[str, Any], model: str | None = None,
                url: str | None = None, key: str | None = None) -> Any:
    """Any server that speaks the OpenAI chat completions API: LM Studio (the default URL), llama.cpp,
    vLLM, OpenRouter, OpenAI itself. CLIPWRIGHT_OPENAI_URL picks the server, OPENAI_API_KEY is sent
    when it's set (local servers don't need one), and the model comes from --copy-model or
    CLIPWRIGHT_OPENAI_MODEL."""
    base = (url or openai_url()).rstrip("/")
    model = model or os.environ.get("CLIPWRIGHT_OPENAI_MODEL")
    if not model:
        raise CopyError("say which model to use: --copy-model NAME or CLIPWRIGHT_OPENAI_MODEL", setup=True)
    key = key if key is not None else os.environ.get("OPENAI_API_KEY")
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": schema}},
        "temperature": 0.7,
    }).encode()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(f"{base}/chat/completions", data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as res:
            out = json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        if e.code in (401, 403):
            raise CopyError(f"{base} refused the key ({e.code}), check OPENAI_API_KEY", setup=True) from None
        if e.code == 404:
            raise CopyError(f"{base} doesn't know model {model!r} ({detail[:120]})", setup=True) from None
        raise CopyError(f"{base} answered {e.code}: {detail[:200]}") from None
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise CopyError(f"can't reach {base} ({e}), is the server running?", setup=True) from None
    try:
        content = out["choices"][0]["message"]["content"]
        return json.loads(content)
    except (KeyError, IndexError, TypeError, ValueError):
        raise CopyError("the answer wasn't the JSON that was asked for") from None


def ask_json(backend: str, prompt: str, system: str, schema: dict[str, Any], model: str | None = None) -> Any:
    """One structured question to any backend. `model` is for ollama and openai (claude uses its own)."""
    if backend == "claude":
        return claude_json(prompt, system, schema)
    if backend == "ollama":
        return ollama_json(prompt, system, schema, model or DEFAULT_OLLAMA_MODEL)
    if backend == "openai":
        return openai_json(prompt, system, schema, model)
    raise ValueError(f"unknown backend {backend!r}")


def write_copy(words: Sequence[Word], backend: str, platform: str | None = None,
               ollama_model: str | None = DEFAULT_OLLAMA_MODEL) -> PostCopy:
    """`ollama_model` is the --copy-model value, used by ollama and openai."""
    if not words:
        raise CopyError("no transcript for this clip")
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}")
    return parse(ask_json(backend, prompt_for(words, platform), SYSTEM, SCHEMA, ollama_model))
