"""llm-based transcript refinement service.

Refines raw ASR (speech-to-text) output by leveraging an LLM to correct
homophones, mis-recognized proper nouns, and domain-specific terminology
that the STT engine got wrong. Two strategies are supported:

- "diff": LLM emits a list of {original, corrected, reason} replacements
  in JSON form. We apply them via word-level matching, preserving the
  original transcript structure (and therefore timestamps / speaker labels).
  This is the SAFER mode and the recommended default.

- "full": LLM rewrites the whole transcript, segment by segment. Higher
  recall but more risk of hallucination/style drift. Use with care.

Backends mirror the SummarizationService split:
- in-process via ``mlx-lm`` / ``mlx-vlm``
- remote via local oMLX server (OpenAI-compatible API)

The actual LLM call helpers (chat-template, thinking-tag stripping) are
imported from ``meetcap.services.summarization`` to avoid duplication.
"""

from __future__ import annotations

import json as _json
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from meetcap.services.summarization import (
    _clean_thinking_tags,
    _strip_untagged_thinking,
)
from meetcap.services.transcription import TranscriptResult, TranscriptSegment

console = Console()

# Chunking limits for the refinement pass. Refinement is per-segment text,
# so we group N segments into a single LLM call rather than chunking by chars.
DEFAULT_SEGMENTS_PER_CHUNK = 80
DEFAULT_MAX_CHARS_PER_CHUNK = 6_000


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Correction:
    """a single LLM-proposed correction applied to the transcript."""

    segment_id: int
    original: str
    corrected: str
    reason: str = ""


@dataclass
class RefinementResult:
    """outcome of a refinement pass."""

    refined_segments: list[TranscriptSegment]
    corrections: list[Correction]
    skipped_chunks: int = 0
    backend: str = ""
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Hotword loading
# ---------------------------------------------------------------------------


def load_hotwords(
    hotwords_inline: list[str] | None = None,
    hotwords_file: Path | None = None,
) -> list[str]:
    """load hotword vocabulary from inline list and/or file.

    File format: plain text, one term per line. Lines starting with ``#`` and
    blank lines are ignored. Inline-list and file entries are merged and
    de-duplicated while preserving first-seen order.
    """
    seen: set[str] = set()
    out: list[str] = []

    def _add(term: str) -> None:
        term = term.strip()
        if not term or term in seen:
            return
        seen.add(term)
        out.append(term)

    if hotwords_inline:
        for w in hotwords_inline:
            _add(w)

    if hotwords_file is not None and hotwords_file.exists():
        try:
            with open(hotwords_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    _add(line)
        except Exception as e:
            console.print(f"[yellow]⚠ failed to read hotwords file {hotwords_file}: {e}[/yellow]")

    return out


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_diff_system_prompt(
    hotwords: list[str],
    meeting_title: str | None,
    preserve_filler_words: bool,
) -> str:
    """system prompt for diff-mode refinement.

    Asks the model to return a strict JSON array of corrections, NOT a
    rewritten transcript.
    """
    hotword_block = (
        "\n".join(f"- {w}" for w in hotwords)
        if hotwords
        else "(no domain-specific terms provided)"
    )
    title_block = f"meeting topic: {meeting_title}\n" if meeting_title else ""
    filler_rule = (
        "preserve filler words (e.g. 'um', 'uh', '嗯', '那个', '就是')."
        if preserve_filler_words
        else "you may quietly drop pure filler words if they break readability."
    )

    return (
        "you are a transcript correction assistant. you receive raw ASR output "
        "from a meeting recording. your job is to identify ONLY clear errors and "
        "propose minimal, surgical replacements.\n\n"
        "STRICT RULES:\n"
        "1. output ONLY a JSON array. no prose, no markdown, no code fences.\n"
        "2. each item has shape: "
        '{\"segment_id\": <int>, \"original\": \"<exact substring>\", '
        '\"corrected\": \"<replacement>\", \"reason\": \"<short>\"}\n'
        "3. \"original\" MUST be an exact substring of the segment text. "
        "do not paraphrase the source.\n"
        "4. fix only: homophones, mis-recognized proper nouns / product names, "
        "obvious typos, and terms from the hotword list below.\n"
        "5. do NOT rewrite sentences. do NOT change style. do NOT add new content.\n"
        f"6. {filler_rule}\n"
        "7. if a segment is already correct, do not emit any item for it.\n"
        "8. if NOTHING needs fixing across the whole input, output exactly: []\n\n"
        f"{title_block}"
        "hotword vocabulary (prefer these spellings when phonetically plausible):\n"
        f"{hotword_block}\n"
    )


def _build_diff_user_prompt(segments: list[TranscriptSegment]) -> str:
    """user prompt: serialize segments as numbered lines."""
    lines = ["transcript segments (segment_id : text):"]
    for s in segments:
        # one line per segment; LLM must reference these ids
        lines.append(f"{s.id}: {s.text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON extraction (robust to LLM noise)
# ---------------------------------------------------------------------------


def _extract_json_array(raw: str) -> list[dict]:
    """pull a JSON array out of LLM output, tolerating prefix/suffix noise.

    Handles common failure modes:
    - leading "Here is the JSON:" prose
    - markdown code fences (```json ... ```)
    - trailing explanatory text
    - empty array
    """
    if not raw or not raw.strip():
        return []

    cleaned = raw.strip()

    # strip code fences
    fence = re.match(r"^```(?:json)?\s*\n(.*?)\n```\s*$", cleaned, re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()

    # try direct parse first
    try:
        parsed = _json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
    except _json.JSONDecodeError:
        pass

    # fall back: locate first '[' and matching ']'
    start = cleaned.find("[")
    if start < 0:
        return []
    depth = 0
    end = -1
    in_str = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        return []

    try:
        parsed = _json.loads(cleaned[start : end + 1])
        if isinstance(parsed, list):
            return parsed
    except _json.JSONDecodeError:
        return []

    return []


# ---------------------------------------------------------------------------
# Correction application
# ---------------------------------------------------------------------------


def _apply_corrections(
    segments: list[TranscriptSegment],
    raw_corrections: list[dict],
) -> tuple[list[TranscriptSegment], list[Correction]]:
    """apply LLM-proposed corrections to segments.

    Returns updated segments (new objects) and the list of actually-applied
    Correction records (skipping ones whose 'original' substring isn't found).
    """
    seg_by_id = {s.id: s for s in segments}
    # mutable copies of text
    new_text: dict[int, str] = {s.id: s.text for s in segments}
    applied: list[Correction] = []

    for item in raw_corrections:
        try:
            seg_id = int(item.get("segment_id"))
            original = str(item.get("original", "")).strip()
            corrected = str(item.get("corrected", "")).strip()
            reason = str(item.get("reason", "")).strip()
        except (TypeError, ValueError):
            continue

        if not original or seg_id not in new_text:
            continue
        if original == corrected:
            continue

        current_text = new_text[seg_id]
        if original not in current_text:
            # try a case-insensitive locator as a softer fallback
            idx = current_text.lower().find(original.lower())
            if idx < 0:
                continue
            real = current_text[idx : idx + len(original)]
            new_text[seg_id] = current_text.replace(real, corrected, 1)
            applied.append(
                Correction(
                    segment_id=seg_id,
                    original=real,
                    corrected=corrected,
                    reason=reason,
                )
            )
            continue

        new_text[seg_id] = current_text.replace(original, corrected, 1)
        applied.append(
            Correction(
                segment_id=seg_id,
                original=original,
                corrected=corrected,
                reason=reason,
            )
        )

    refined: list[TranscriptSegment] = []
    for s in segments:
        if new_text[s.id] != s.text:
            refined.append(
                TranscriptSegment(
                    id=s.id,
                    start=s.start,
                    end=s.end,
                    text=new_text[s.id],
                    speaker_id=s.speaker_id,
                    confidence=s.confidence,
                )
            )
        else:
            # keep original object reference for unchanged segments
            refined.append(seg_by_id[s.id])

    return refined, applied


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _chunk_segments(
    segments: list[TranscriptSegment],
    segs_per_chunk: int = DEFAULT_SEGMENTS_PER_CHUNK,
    max_chars: int = DEFAULT_MAX_CHARS_PER_CHUNK,
) -> list[list[TranscriptSegment]]:
    """split segments into chunks bounded by both segment count and char count."""
    chunks: list[list[TranscriptSegment]] = []
    current: list[TranscriptSegment] = []
    current_chars = 0
    for s in segments:
        seg_chars = len(s.text) + 8  # rough overhead per line
        if current and (
            len(current) >= segs_per_chunk or current_chars + seg_chars > max_chars
        ):
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(s)
        current_chars += seg_chars
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Refinement service
# ---------------------------------------------------------------------------


class RefinementService:
    """LLM-driven post-ASR transcript correction.

    Call ``refine(transcript_result)`` to get a ``RefinementResult``.

    Backends:
        - "mlx-lm" (default): in-process inference via mlx-lm/mlx-vlm
        - "omlx": HTTP call to a local oMLX OpenAI-compatible server

    Both backends share the same prompts and JSON parsing logic.
    """

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen3.5-2B-OptiQ-4bit",
        backend: str = "mlx-lm",
        mode: str = "diff",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        hotwords: list[str] | None = None,
        preserve_filler_words: bool = True,
        meeting_title: str | None = None,
        # oMLX-specific
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "",
        timeout: int = 300,
        # chunking
        segs_per_chunk: int = DEFAULT_SEGMENTS_PER_CHUNK,
        max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
    ):
        if mode not in ("diff", "full"):
            raise ValueError(f"unsupported refinement mode: {mode!r}")
        if backend not in ("mlx-lm", "omlx"):
            raise ValueError(f"unsupported refinement backend: {backend!r}")

        self.model_name = model_name
        self.backend = backend
        self.mode = mode
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.hotwords = hotwords or []
        self.preserve_filler_words = preserve_filler_words
        self.meeting_title = meeting_title
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.segs_per_chunk = segs_per_chunk
        self.max_chars_per_chunk = max_chars_per_chunk

        # mlx-lm in-process state (lazily loaded)
        self._model = None
        self._processor = None
        self._mlx_backend: str | None = None  # "mlx-lm" | "mlx-vlm"
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------ load

    def load_model(self) -> None:
        """eagerly load the in-process model (no-op for oMLX)."""
        if self.backend == "omlx":
            return
        self._load_in_process()

    def _load_in_process(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            console.print(f"[cyan]loading refinement model {self.model_name}...[/cyan]")
            try:
                from mlx_lm import load

                self._model, self._processor = load(self.model_name)
                self._mlx_backend = "mlx-lm"
                return
            except (ImportError, Exception):
                pass

            try:
                from mlx_vlm import load
                from mlx_vlm.utils import load_config  # noqa: F401

                self._model, self._processor = load(self.model_name)
                self._mlx_backend = "mlx-vlm"
            except ImportError as e:
                raise ImportError(
                    "neither mlx-lm nor mlx-vlm installed. install with: pip install mlx-lm"
                ) from e

    def unload_model(self) -> None:
        """release the in-process model (no-op for oMLX)."""
        if self.backend == "omlx":
            console.print("[dim]omlx manages refinement model lifecycle externally[/dim]")
            return
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        import gc

        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except (ImportError, Exception):
            pass
        console.print("[dim]refinement model unloaded[/dim]")

    def is_loaded(self) -> bool:
        if self.backend == "omlx":
            return self._omlx_reachable()
        return self._model is not None

    # ---------------------------------------------------------------- public

    def refine(self, transcript: TranscriptResult) -> RefinementResult:
        """refine a transcription result.

        Mutates nothing — returns a new RefinementResult with refined segments
        and a list of applied corrections.
        """
        if not transcript.segments:
            return RefinementResult(
                refined_segments=[],
                corrections=[],
                backend=self.backend,
            )

        start = time.time()
        chunks = _chunk_segments(
            transcript.segments,
            segs_per_chunk=self.segs_per_chunk,
            max_chars=self.max_chars_per_chunk,
        )

        all_corrections_raw: list[dict] = []
        skipped = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(
                f"refining transcript ({len(chunks)} chunk(s), {self.backend})...",
                total=len(chunks),
            )
            for i, chunk in enumerate(chunks):
                try:
                    raw_items = self._refine_chunk(chunk)
                    all_corrections_raw.extend(raw_items)
                except Exception as e:
                    skipped += 1
                    console.print(
                        f"[yellow]⚠ refinement chunk {i + 1}/{len(chunks)} failed: {e}[/yellow]"
                    )
                progress.update(task, advance=1)

        refined_segments, applied = _apply_corrections(
            transcript.segments, all_corrections_raw
        )

        duration = time.time() - start
        if applied:
            console.print(
                f"[green]✓[/green] refinement applied {len(applied)} correction(s) "
                f"in {duration:.1f}s"
                + (f" ({skipped} chunk(s) skipped)" if skipped else "")
            )
        else:
            console.print(
                f"[dim]refinement found nothing to fix ({duration:.1f}s)"
                + (f", {skipped} chunk(s) skipped" if skipped else "")
                + "[/dim]"
            )

        return RefinementResult(
            refined_segments=refined_segments,
            corrections=applied,
            skipped_chunks=skipped,
            backend=self.backend,
            duration_seconds=duration,
        )

    # ------------------------------------------------------------- internals

    def _refine_chunk(self, segments: list[TranscriptSegment]) -> list[dict]:
        """run one LLM call on one chunk; return parsed JSON items."""
        system_prompt = _build_diff_system_prompt(
            self.hotwords, self.meeting_title, self.preserve_filler_words
        )
        user_prompt = _build_diff_user_prompt(segments)

        if self.backend == "omlx":
            raw = self._call_omlx(system_prompt, user_prompt)
        else:
            raw = self._call_in_process(system_prompt, user_prompt)

        return _extract_json_array(raw)

    # --- in-process backend (mlx-lm / mlx-vlm) ---

    def _call_in_process(self, system_prompt: str, user_prompt: str) -> str:
        self._load_in_process()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        if self._mlx_backend == "mlx-lm":
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(temp=self.temperature)
            raw = generate(
                self._model,
                self._processor,
                prompt=prompt,
                max_tokens=self.max_tokens,
                sampler=sampler,
            )
        else:  # mlx-vlm
            from mlx_vlm import generate

            result = generate(
                self._model,
                self._processor,
                prompt,
                max_tokens=self.max_tokens,
                temp=self.temperature,
            )
            if hasattr(result, "text"):
                raw = result.text
            elif isinstance(result, str):
                raw = result
            else:
                raw = str(result)

        return _post_process_llm_output(raw)

    # --- oMLX backend ---

    def _omlx_reachable(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/models")
            if self.api_key:
                req.add_header("Authorization", f"Bearer {self.api_key}")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _call_omlx(self, system_prompt: str, user_prompt: str) -> str:
        if not self._omlx_reachable():
            raise ConnectionError(
                f"oMLX server not reachable at {self.base_url}. "
                "start it with: brew services start omlx  (or: omlx serve)"
            )

        payload: dict = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        data = _json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = _json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            raise RuntimeError(
                f"oMLX API error {e.code}: {error_body[:500]}"
            ) from e
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"cannot connect to oMLX at {self.base_url}: {e.reason}"
            ) from e

        raw = result["choices"][0]["message"]["content"]
        return _post_process_llm_output(raw)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _post_process_llm_output(text: str) -> str:
    """strip thinking tags / preamble from LLM output before JSON parsing."""
    if not text:
        return ""
    text = text.strip()
    if "<think" in text.lower() or "</think" in text.lower():
        text = _clean_thinking_tags(text)
    text = _strip_untagged_thinking(text)
    return text


def save_corrections_log(
    corrections: list[Correction],
    base_path: Path,
    metadata: dict | None = None,
) -> Path:
    """write the audit log of applied corrections to disk."""
    log_path = base_path.with_suffix(".corrections.json")
    payload = {
        "metadata": metadata or {},
        "count": len(corrections),
        "corrections": [asdict(c) for c in corrections],
    }
    with open(log_path, "w", encoding="utf-8") as f:
        _json.dump(payload, f, indent=2, ensure_ascii=False)
    console.print(f"[green]✓[/green] corrections log saved: {log_path}")
    return log_path


def backup_original_transcript(base_path: Path) -> tuple[Path | None, Path | None]:
    """copy the existing transcript text+json to .original.* siblings.

    Returns the paths of the backups created (or None if originals didn't exist).
    Used before overwriting recording.transcript.{txt,json} with the refined
    version, so users can always diff or revert.
    """
    import shutil

    txt = base_path.with_suffix(".transcript.txt")
    js = base_path.with_suffix(".transcript.json")
    txt_backup = base_path.with_suffix(".transcript.original.txt")
    js_backup = base_path.with_suffix(".transcript.original.json")

    txt_out: Path | None = None
    js_out: Path | None = None

    if txt.exists() and not txt_backup.exists():
        shutil.copy2(txt, txt_backup)
        txt_out = txt_backup
    if js.exists() and not js_backup.exists():
        shutil.copy2(js, js_backup)
        js_out = js_backup

    return txt_out, js_out
