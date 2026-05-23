# Transcript Refinement (LLM Post-Editing)

> Status: implemented in branch `feat/llm-transcript-refinement`.
> Module: `meetcap/services/refinement.py`
> CLI: `meetcap refine` + automatic stage in the record / summarize / reprocess pipelines.

## 1、Why post-edit instead of ASR hotwords?

Most production speech recognizers (whisper.cpp, mlx-whisper, faster-whisper, parakeet, vosk) either don't expose a hotword / contextual biasing API at all, or expose one that requires re-running inference and only weakly biases the language model. For a meeting recorder that has to support **multiple STT backends** and recover from "TCADP → 提卡 ADP" or "Cognito → 我看你头" style mistakes, post-editing the transcript with a local LLM is:

- **Backend-agnostic** — the same fix works whether the user picked mlx-whisper or faster-whisper.
- **Cheap** — runs on text only, no second round of audio inference.
- **Iterable** — change the hotword list and re-run `meetcap refine`; no need to redo the recording.
- **Auditable** — every edit is logged with a reason in `corrections.json`.

The trade-off is hallucination risk. We mitigate it by:

1. Defaulting to **diff mode** — the LLM proposes per-segment substring edits, not a wholesale rewrite.
2. Using `temperature = 0.0`.
3. Asking for **strict JSON output** and silently dropping malformed responses.
4. Keeping a **backup** of the original transcript so the user can always revert.

## 2、Architecture

```
┌──────────────┐    ┌─────────────────────┐    ┌──────────────────────┐
│   STT        │───▶│ TranscriptResult    │───▶│ RefinementService    │
│ (any engine) │    │ segments[*].text    │    │ ─ chunk segments     │
└──────────────┘    │ segments[*].speaker │    │ ─ build prompt       │
                    │ segments[*].start   │    │ ─ call LLM (mlx-lm   │
                    │ segments[*].end     │    │   or oMLX HTTP)      │
                    └─────────────────────┘    │ ─ parse JSON array   │
                                               │ ─ apply substring    │
                                               │   replacements       │
                                               └─────────┬────────────┘
                                                         ▼
                                  ┌────────────────────────────────────┐
                                  │ recording.transcript.{txt,json}    │ (overwritten)
                                  │ recording.transcript.original.*    │ (backup)
                                  │ recording.corrections.json         │ (audit log)
                                  └────────────────────────────────────┘
```

Refinement happens **after** STT model unload (to free VRAM) and **before** summarization, so downstream stages consume the corrected text.

## 3、Operating modes

### 3.1 Diff mode (default, recommended)

The LLM is asked to return a JSON array such as:

```json
[
  {"seg_index": 3, "original": "提卡 ADP", "corrected": "TCADP", "reason": "hotword: TCADP"},
  {"seg_index": 7, "original": "Coginto", "corrected": "Cognito", "reason": "hotword: Cognito"}
]
```

For each entry, we do a **word-level substring replacement** in `segments[seg_index].text`. Timestamps, speaker labels, and segment boundaries are untouched. Filler words (`um`, `uh`, `嗯`) are kept by default to preserve fidelity.

If a chunk fails (network error, malformed JSON after fallback parsing), we record it in `RefinementResult.skipped_chunks` and continue with the next chunk — partial improvement beats no improvement.

### 3.2 Full mode (experimental)

Sends the whole chunk to the LLM and asks for the rewritten text. **Higher hallucination risk**, only use when the transcript needs structural cleanup beyond term substitution. Currently exposed via config but the recommended path remains `mode = "diff"`.

## 4、Chunking algorithm

`_chunk_segments(segments, segs_per_chunk, max_chars_per_chunk)` splits the transcript so each LLM call sees a coherent block:

- Hard limit: `segs_per_chunk` segments per chunk (default 12).
- Soft limit: `max_chars_per_chunk` characters of cumulative text (default 3000) — flush early when exceeded.
- Speaker turns are never broken mid-segment.
- Each segment in the prompt is prefixed with its **original `seg_index`** so the LLM can reference it in the JSON output even though its position inside the chunk is local.

## 5、Prompt design

System prompt (diff mode, simplified):

```
You are a transcript correction assistant. The user provides ASR-generated meeting
segments and a list of hotwords (proper nouns, acronyms, jargon). Your job is to
identify mis-transcribed substrings and propose minimal corrections.

Rules:
- Output ONLY a JSON array. No prose, no code fences.
- Each element: {"seg_index": int, "original": str, "corrected": str, "reason": str}
- Only correct clear ASR errors. Do NOT rewrite for style.
- Preserve filler words (um, uh, 嗯) unless preserve_filler_words=false.
- If nothing needs fixing, return [].
```

User prompt includes:

1. The hotword list (deduplicated, one per line).
2. Optional meeting title for context.
3. The chunk's segments, formatted as `[seg_index] (speaker) text`.

`_extract_json_array()` is intentionally lenient: it strips ```json fences, tolerates leading/trailing prose, and uses bracket-depth counting to find the JSON array even when the LLM adds explanations.

## 6、Configuration

See README "Transcript Refinement" section for the full `[refinement]` block. Key fields:

| Field | Default | Notes |
|---|---|---|
| `enabled` | `false` | Master switch for the in-pipeline stage. CLI `meetcap refine` ignores this. |
| `mode` | `"diff"` | `"diff"` or `"full"`. |
| `backend` | `""` | `"mlx-lm"` (in-process) or `"omlx"` (HTTP). Empty inherits from `[llm].backend`. |
| `model_name` | `""` | LLM model id / path. Empty inherits from `[llm]`. |
| `temperature` | `0.0` | Keep low to reduce hallucination. |
| `max_tokens` | `1024` | Per-chunk output budget. |
| `preserve_filler_words` | `true` | Keep um/uh/嗯. |
| `hotwords_file` | `~/.meetcap/hotwords.txt` | One term per line, `#` comments allowed. |
| `hotwords` | `[]` | Inline list, **merged** with file. |
| `segs_per_chunk` | `12` | |
| `max_chars_per_chunk` | `3000` | |
| `keep_original` | `true` | Save `.transcript.original.{txt,json}` before overwriting. |

### Environment variables

All fields are also exposed as `MEETCAP_REFINEMENT_*` environment variables (uppercase, dot → underscore), e.g. `MEETCAP_REFINEMENT_ENABLED=1`, `MEETCAP_REFINEMENT_HOTWORDS_FILE=/path/to/hotwords.txt`.

## 7、CLI: `meetcap refine`

```bash
meetcap refine RECORDING_DIR [options]

Options:
  --hotwords-file PATH   Override [refinement].hotwords_file
  --backend TEXT         "mlx-lm" or "omlx"
  --llm TEXT             LLM model name / path
  --mode TEXT            "diff" or "full"
  --log-file PATH        Custom corrections log path (default: <dir>/recording.corrections.json)
  --yes / -y             Skip confirmation prompt
```

`RECORDING_DIR` may be either an absolute path or a folder name under `[paths].output_dir`.

The command:

1. Loads `recording.transcript.json` from the directory.
2. Backs up to `recording.transcript.original.{txt,json}` if `keep_original`.
3. Calls `RefinementService.refine()`.
4. Rewrites `recording.transcript.{txt,json}`.
5. Writes `recording.corrections.json`.
6. Prints a summary table: `corrections applied`, `chunks processed`, `chunks skipped`.

## 8、Hotwords file format

Plain UTF-8 text. One term per line. Whitespace trimmed. Lines starting with `#` and blank lines are ignored. Duplicates collapsed.

```
# Tencent Cloud
TCADP
TKE
COS
CDB

# Customer names
Jucoin
PT Cyrameta

# People
Andrej Karpathy

# Acronyms the ASR keeps splitting
LLM
ASR
RAG
```

A starter template ships at `~/.meetcap/hotwords.txt` (created by `meetcap setup` and also placed in `assets/hotwords.example.txt` in the repo).

## 9、Testing

`tests/test_refinement.py` covers:

- Hotword loading (file, inline, dedup, comments).
- JSON extraction (clean, fenced, prose-wrapped, malformed).
- Substring application (single, multi-occurrence, cross-segment safety).
- Chunking (segment limit, char limit, speaker boundaries).
- Mlx-lm backend (mocked).
- Persistence (backup + audit log).
- Config defaults & env var overrides.

42 tests, all passing on the dev machine. Backend-specific paths (real mlx-lm and oMLX inference) are mocked; integration testing happens manually against a recording.

## 10、Known limitations & future work

- **LLM still can be wrong.** Always glance at `corrections.json` for a sample, especially the first time you add a new hotword.
- **No fuzzy matching beyond the LLM.** We don't pre-filter chunks — every chunk is sent. Skip-on-no-hotword-hit is a possible optimization.
- **Diff mode assumes the LLM picks the right `seg_index`.** If it confuses indices, the substring replace simply finds nothing and the edit is a no-op (safe failure).
- **Full mode is unfinished.** Treat as experimental.
- **No diarization-aware merging.** Adjacent same-speaker segments aren't joined before refinement; this can occasionally split a hotword across two segments. Mitigation: enlarge `segs_per_chunk` so the LLM at least sees both halves.

## 11、Related files

- `meetcap/services/refinement.py` — service implementation.
- `meetcap/utils/config.py` — `[refinement]` section + env var mapping.
- `meetcap/cli.py` — `_maybe_refine_transcript()` pipeline hook + `refine` command.
- `tests/test_refinement.py` — unit tests.
- `assets/hotwords.example.txt` — starter hotword list (English + Chinese examples).
