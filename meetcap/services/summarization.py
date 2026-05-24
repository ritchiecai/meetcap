"""llm-based meeting summarization using qwen3.5 via mlx-lm, mlx-vlm, or omlx server"""

import json as _json
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# fixed chunking constants (chars, not tokens)
CHUNK_THRESHOLD = 500_000
CHUNK_SIZE = 400_000


# ---------------------------------------------------------------------------
# Shared helpers (used by both SummarizationService and OmlxSummarizationService)
# ---------------------------------------------------------------------------


def _build_system_prompt(has_speaker_info: bool) -> str:
    """build the system prompt for meeting summarization."""
    if has_speaker_info:
        return (
            "you are a world-class executive assistant who creates comprehensive, actionable meeting summaries. "
            "the transcript includes speaker labels (e.g., [Speaker 1], [Speaker 2]). "
            "analyze the transcript and produce detailed notes with these exact sections:\n\n"
            "## Meeting Title\n"
            "generate a concise title (2-4 words) that captures the main topic of the meeting.\n"
            "the title should be in PascalCase with no spaces (e.g., 'ProductRoadmap', 'TeamRetrospective').\n"
            "write only the title on a single line, nothing else in this section.\n\n"
            "## Summary\n"
            "provide a comprehensive 3-5 paragraph summary covering:\n"
            "- main topics discussed and context\n"
            "- key points raised by specific speakers\n"
            "- important details, data, or examples mentioned\n"
            "- overall meeting outcome and next steps\n\n"
            "## Participants\n"
            "list the speakers identified in the transcript:\n"
            "- note their key contributions or roles in the discussion\n"
            "- identify who led the meeting if apparent\n\n"
            "## Key Discussion Points\n"
            "list 5-10 bullet points of the most important topics discussed:\n"
            "- include specific details and context for each point\n"
            "- attribute key points to specific speakers when relevant\n"
            "- note any disagreements or alternative viewpoints between speakers\n"
            "- highlight critical information or insights shared\n\n"
            "## Decisions Made\n"
            "list all decisions made during the meeting:\n"
            "- be specific about what was decided\n"
            "- include rationale if discussed\n"
            "- note which speaker made or supported the decision\n"
            "- if no decisions were made, write 'no formal decisions made'\n\n"
            "## Action Items\n"
            "list all tasks and follow-ups mentioned:\n"
            "- format: - [ ] owner (Speaker X) — detailed task description (due: yyyy-mm-dd)\n"
            "- if no owner mentioned, use 'tbd' as owner\n"
            "- if no date mentioned, use 'tbd' for date\n"
            "- include context for why each action is needed\n"
            "- if no action items, write 'no action items identified'\n\n"
            "## Notable Quotes\n"
            "include 2-3 important verbatim quotes with speaker attribution\n\n"
            "## Meeting Tone\n"
            "briefly describe the overall tone and energy of the meeting.\n\n"
            "IMPORTANT GUIDELINES:\n"
            "- DO NOT attempt to identify or include the actual names of meeting participants. "
            "The transcription system is unreliable with names, so refer to speakers only by their labels (e.g., 'Speaker 1', 'Speaker 2').\n"
            "- DO NOT expand acronyms or assume what they mean. Write acronyms exactly as spoken (e.g., write 'API' not 'Application Programming Interface'). "
            "The summarization process tends to make errors when expanding abbreviations.\n\n"
            "be thorough and detailed while maintaining clarity. "
            "do not include any thinking tags or meta-commentary."
        )
    else:
        return (
            "you are a world-class executive assistant who creates comprehensive, actionable meeting summaries. "
            "analyze the transcript and produce detailed notes with these exact sections:\n\n"
            "## Meeting Title\n"
            "generate a concise title (2-4 words) that captures the main topic of the meeting.\n"
            "the title should be in PascalCase with no spaces (e.g., 'ProductRoadmap', 'TeamRetrospective').\n"
            "write only the title on a single line, nothing else in this section.\n\n"
            "## Summary\n"
            "provide a comprehensive 3-5 paragraph summary covering:\n"
            "- main topics discussed and context\n"
            "- key points raised by participants\n"
            "- important details, data, or examples mentioned\n"
            "- overall meeting outcome and next steps\n\n"
            "## Participants\n"
            "list any participants mentioned or implied in the transcript.\n\n"
            "## Key Discussion Points\n"
            "list 5-10 bullet points of the most important topics discussed:\n"
            "- include specific details and context for each point\n"
            "- note any disagreements or alternative viewpoints\n"
            "- highlight critical information or insights shared\n\n"
            "## Decisions Made\n"
            "list all decisions made during the meeting:\n"
            "- be specific about what was decided\n"
            "- include rationale if discussed\n"
            "- note who made or supported the decision\n"
            "- if no decisions were made, write 'no formal decisions made'\n\n"
            "## Action Items\n"
            "list all tasks and follow-ups mentioned:\n"
            "- format: - [ ] owner — detailed task description (due: yyyy-mm-dd)\n"
            "- if no owner mentioned, use 'tbd' as owner\n"
            "- if no date mentioned, use 'tbd' for date\n"
            "- include context for why each action is needed\n"
            "- if no action items, write 'no action items identified'\n\n"
            "## Notable Quotes\n"
            "include 2-3 important verbatim quotes that capture key insights or decisions\n\n"
            "## Meeting Tone\n"
            "briefly describe the overall tone and energy of the meeting.\n\n"
            "IMPORTANT GUIDELINES:\n"
            "- DO NOT attempt to identify or include the actual names of meeting participants. "
            "The transcription system is unreliable with names, so refer to participants generically (e.g., 'one participant mentioned', 'a team member noted').\n"
            "- DO NOT expand acronyms or assume what they mean. Write acronyms exactly as spoken (e.g., write 'API' not 'Application Programming Interface'). "
            "The summarization process tends to make errors when expanding abbreviations.\n\n"
            "be thorough and detailed while maintaining clarity. "
            "do not include any thinking tags or meta-commentary."
        )


def _build_user_prompt(
    transcript_text: str,
    meeting_title: str | None = None,
    attendees: list[str] | None = None,
    manual_notes_path: Path | None = None,
) -> str:
    """build the user prompt including transcript and optional context."""
    user_prompt_parts = []

    if meeting_title:
        user_prompt_parts.append(f"meeting: {meeting_title}")
    if attendees:
        user_prompt_parts.append(f"attendees: {', '.join(attendees)}")

    # add manual notes if available
    if manual_notes_path and manual_notes_path.exists():
        try:
            with open(manual_notes_path, encoding="utf-8") as f:
                manual_notes_text = f.read()
            if manual_notes_text.strip():
                console.print("[dim]manual notes found, including in summary[/dim]")
                user_prompt_parts.append(f"manual notes:\n{manual_notes_text}")
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] could not read manual notes: {e}")

    user_prompt_parts.append(f"transcript:\n{transcript_text}")
    return "\n\n".join(user_prompt_parts)


def _chunk_transcript(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """
    split transcript into chunks for processing.

    args:
        text: full transcript text
        chunk_size: approximate size of each chunk in chars

    returns:
        list of text chunks
    """
    chunks: list[str] = []
    words = text.split()
    current_chunk: list[str] = []
    current_size = 0

    for word in words:
        word_len = len(word) + 1  # +1 for space
        if current_size + word_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = word_len
        else:
            current_chunk.append(word)
            current_size += word_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def _clean_thinking_tags(text: str) -> str:
    """
    remove thinking tags from llm output.

    qwen3-thinking models include <think>...</think> tags that should be removed.
    handles edge cases like malformed tags, missing opening tags, etc.

    args:
        text: raw llm output possibly containing thinking tags

    returns:
        cleaned text without thinking tags
    """
    # handle case where content appears before </think> without opening tag
    if "</think>" in text.lower() and "<think" not in text.lower():
        pattern = r"(?is)^.*?</think\s*>"
        cleaned = re.sub(pattern, "", text)
    else:
        cleaned = re.sub(r"(?is)<think[^>]*?>.*?</think\s*>", "", text)

    # also handle <thinking>...</thinking> variant
    if "</thinking>" in cleaned.lower() and "<thinking" not in cleaned.lower():
        pattern = r"(?is)^.*?</thinking\s*>"
        cleaned = re.sub(pattern, "", cleaned)
    else:
        cleaned = re.sub(r"(?is)<thinking[^>]*?>.*?</thinking\s*>", "", cleaned)

    # remove any remaining lone tags (opening or closing)
    cleaned = re.sub(r"(?i)</?think[^>]*?>", "", cleaned)
    cleaned = re.sub(r"(?i)</?thinking[^>]*?>", "", cleaned)

    # clean up any extra whitespace that may be left
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)  # collapse multiple newlines
    cleaned = re.sub(r"^\n+", "", cleaned)  # remove leading newlines

    return cleaned.strip()


def _strip_untagged_thinking(text: str) -> str:
    """
    remove untagged thinking content that some models emit without <think> tags.

    detects patterns like "Thinking Process:" or "Thinking:" at the start of output
    followed by the actual summary content starting with a markdown heading (# or ##).

    args:
        text: output text possibly containing untagged thinking

    returns:
        cleaned text with only the actual summary content
    """
    # pattern: text starts with thinking-like header, actual content starts at first markdown heading
    thinking_prefixes = (
        "thinking process:",
        "thinking:",
        "my thinking:",
        "internal reasoning:",
        "reasoning:",
        "let me think",
        "i need to",
        "1.  **analyze",
        "1. **analyze",
    )

    text_lower = text.lower().lstrip()
    if any(text_lower.startswith(prefix) for prefix in thinking_prefixes):
        # find the first markdown heading that looks like actual summary content
        # look for ## Meeting Title, ## Summary, # Meeting, etc.
        match = re.search(r"^(#{1,2}\s+(?:meeting|summary|key|participants|decisions|action|notable))", text, re.MULTILINE | re.IGNORECASE)
        if match:
            return text[match.start():].strip()

    # also handle case where thinking appears as numbered analysis before actual content
    # e.g., "1. **Analyze the Request:**\n..." followed by actual markdown
    if text_lower.startswith("1.") and "**" in text[:50]:
        match = re.search(r"^(#{1,2}\s+(?:meeting|summary|key|participants|decisions|action|notable))", text, re.MULTILINE | re.IGNORECASE)
        if match:
            return text[match.start():].strip()

    return text


# ---------------------------------------------------------------------------
# OmlxSummarizationService — calls local oMLX server via OpenAI-compatible API
# ---------------------------------------------------------------------------


class OmlxSummarizationService:
    """generate meeting summaries via local oMLX server (OpenAI-compatible API).

    oMLX (https://github.com/jundot/omlx) provides continuous batching, tiered
    KV caching (RAM + SSD), and multi-model serving on Apple Silicon. This service
    offloads LLM inference to the running oMLX server, freeing meetcap's process
    memory for STT and diarization workloads.

    requirements:
        - oMLX running locally (brew services start omlx, or omlx serve)
        - target model loaded in oMLX (auto-managed via LRU)
    """

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen3.5-2B-OptiQ-4bit",
        base_url: str = "http://localhost:8000/v1",
        temperature: float = 0.4,
        max_tokens: int = 4096,
        enable_thinking: bool = False,
        thinking_budget: int = 512,
        api_key: str = "",
        timeout: int = 300,
    ):
        """
        initialize omlx summarization service.

        args:
            model_name: model name as known to oMLX (HuggingFace repo id or local name)
            base_url: oMLX API base URL (default: http://localhost:8000/v1)
            temperature: sampling temperature (0.2-0.6 recommended)
            max_tokens: max tokens to generate
            enable_thinking: enable thinking mode for the model
            thinking_budget: max tokens for thinking block (when enabled)
            api_key: API key for oMLX (empty string if no auth configured)
            timeout: HTTP request timeout in seconds (default: 300s for long transcripts)
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.api_key = api_key
        self.timeout = timeout

    def load_model(self) -> None:
        """no-op: oMLX manages model loading automatically."""
        pass

    def unload_model(self) -> None:
        """no-op: oMLX manages model lifecycle via LRU eviction."""
        console.print("[dim]omlx manages model lifecycle externally[/dim]")

    def is_loaded(self) -> bool:
        """check if oMLX server is reachable and model is available."""
        try:
            req = urllib.request.Request(f"{self.base_url}/models")
            if self.api_key:
                req.add_header("Authorization", f"Bearer {self.api_key}")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    data = _json.loads(resp.read().decode("utf-8"))
                    models = [m.get("id", "") for m in data.get("data", [])]
                    # check if our model (or a matching suffix) is available
                    model_short = self.model_name.split("/")[-1] if "/" in self.model_name else self.model_name
                    return any(model_short in m or self.model_name in m for m in models)
            return False
        except Exception:
            return False

    def summarize(
        self,
        transcript_text: str,
        meeting_title: str | None = None,
        attendees: list[str] | None = None,
        has_speaker_info: bool = False,
        manual_notes_path: Path | None = None,
    ) -> str:
        """
        generate meeting summary from transcript via oMLX.

        args:
            transcript_text: full transcript text
            meeting_title: optional meeting title
            attendees: optional list of attendees
            has_speaker_info: whether transcript includes speaker labels
            manual_notes_path: optional path to user's manual notes file

        returns:
            markdown-formatted summary
        """
        # verify server is reachable
        if not self._check_server():
            raise ConnectionError(
                f"oMLX server not reachable at {self.base_url}. "
                "start it with: brew services start omlx  (or: omlx serve)"
            )

        console.print(f"[cyan]generating meeting summary via oMLX ({self.model_name})...[/cyan]")
        start_time = time.time()

        system_prompt = _build_system_prompt(has_speaker_info)
        user_prompt = _build_user_prompt(
            transcript_text, meeting_title, attendees, manual_notes_path
        )

        # chunk if needed
        if len(user_prompt) > CHUNK_THRESHOLD:
            chunks = _chunk_transcript(transcript_text, chunk_size=CHUNK_SIZE)

            summaries = []
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task(
                    f"processing {len(chunks)} chunks via oMLX...", total=len(chunks)
                )
                for i, chunk in enumerate(chunks):
                    chunk_prompt = f"transcript chunk {i + 1}/{len(chunks)}:\n{chunk}"
                    summary = self._call_omlx(system_prompt, chunk_prompt)
                    summaries.append(summary)
                    progress.update(task, advance=1)

            # merge summaries
            if len(summaries) > 1:
                merge_prompt = (
                    "merge these partial summaries into one final summary:\n\n"
                    + "\n---\n".join(summaries)
                )
                final_summary = self._call_omlx(system_prompt, merge_prompt)
            else:
                final_summary = summaries[0]
        else:
            # single pass for short transcripts
            with Progress(
                SpinnerColumn(),
                TextColumn("generating summary via oMLX..."),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("", total=None)
                final_summary = self._call_omlx(system_prompt, user_prompt)

        duration = time.time() - start_time
        console.print(f"[green]✓[/green] summary generated in {duration:.1f}s (oMLX)")

        return final_summary

    def _check_server(self) -> bool:
        """quick health check on oMLX server."""
        try:
            req = urllib.request.Request(f"{self.base_url}/models")
            if self.api_key:
                req.add_header("Authorization", f"Bearer {self.api_key}")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _call_omlx(self, system_prompt: str, user_prompt: str) -> str:
        """
        make a single chat completion call to oMLX.

        args:
            system_prompt: system instructions
            user_prompt: user input with transcript

        returns:
            generated text from the model
        """
        payload: dict = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "top_p": 0.95,
            "max_tokens": self.max_tokens,
            "stream": False,
            # tokens that should hard-stop generation (defensive against
            # chat-template token leaks from server-side rendering)
            "stop": ["<|im_end|>", "<|endoftext|>"],
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }

        # if thinking is enabled, pass thinking_budget
        if self.enable_thinking:
            payload["chat_template_kwargs"]["thinking_budget"] = self.thinking_budget

        data = _json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
            },
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

        # defensive response parsing: oMLX (and OpenAI-compatible servers in
        # general) may put the actual answer in `reasoning_content` instead of
        # `content` when the model emits thinking tokens, or return malformed
        # payloads under high load. fail loudly with diagnostic info instead
        # of silently feeding `None` into the post-processing pipeline.
        choices = result.get("choices") or []
        if not choices:
            raise RuntimeError(
                f"oMLX returned no choices; raw response keys={list(result.keys())}"
            )
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            # fall back to reasoning_content if the server split the output
            reasoning = message.get("reasoning_content")
            if reasoning:
                console.print(
                    "[dim]oMLX returned empty content; using reasoning_content[/dim]"
                )
                content = reasoning
        if not content:
            raise RuntimeError(
                "oMLX returned empty content; "
                f"message keys={list(message.keys())}, "
                f"finish_reason={choices[0].get('finish_reason')!r}"
            )

        raw_output = content.strip()

        # always clean thinking tags — models may output them regardless of enable_thinking
        if "<think" in raw_output.lower() or "</think" in raw_output.lower():
            console.print("[dim]detected thinking tags in output, cleaning...[/dim]")
            raw_output = _clean_thinking_tags(raw_output)

        # handle untagged thinking content (e.g., "Thinking Process:" prefix)
        raw_output = _strip_untagged_thinking(raw_output)

        # warn if output seems empty
        if not raw_output or len(raw_output) < 10:
            console.print("[yellow]⚠ output seems very short after cleaning[/yellow]")

        return raw_output


# ---------------------------------------------------------------------------
# SummarizationService — in-process mlx-lm / mlx-vlm inference (original)
# ---------------------------------------------------------------------------


class SummarizationService:
    """generate meeting summaries using local llm via mlx-lm (or mlx-vlm fallback)"""

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen3.5-2B-OptiQ-4bit",
        temperature: float = 0.4,
        max_tokens: int = 4096,
        enable_thinking: bool = False,
        thinking_budget: int = 512,
    ):
        """
        initialize summarization service.

        args:
            model_name: huggingface repo id for the mlx model
            temperature: sampling temperature (0.2-0.6 recommended)
            max_tokens: max tokens to generate
            enable_thinking: enable thinking mode (model reasons before answering)
            thinking_budget: max tokens for the thinking block (when enabled)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.model = None
        self.processor = None
        self.model_config = None
        self._backend = None  # "mlx-lm" or "mlx-vlm"
        self._load_lock = threading.Lock()

    def _load_model(self) -> None:
        """lazy load the llm model."""
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:
                return  # another thread loaded while we waited

            console.print(f"[cyan]loading llm model {self.model_name}...[/cyan]")

            # prefer mlx-lm (works with OptiQ and standard text models)
            try:
                from mlx_lm import load

                self.model, self.processor = load(self.model_name)
                self._backend = "mlx-lm"
                return
            except (ImportError, Exception):
                pass

            # fall back to mlx-vlm (for vision-language models)
            try:
                from mlx_vlm import load
                from mlx_vlm.utils import load_config

                self.model_config = load_config(self.model_name)
                self.model, self.processor = load(self.model_name)
                self._backend = "mlx-vlm"
            except ImportError as e:
                raise ImportError(
                    "neither mlx-lm nor mlx-vlm installed. install with: pip install mlx-lm"
                ) from e

    def load_model(self) -> None:
        """explicitly load the llm model."""
        self._load_model()

    def unload_model(self) -> None:
        """unload mlx-vlm model and cleanup resources."""
        if hasattr(self, "model") and self.model is not None:
            del self.model
        self.model = None
        if hasattr(self, "processor") and self.processor is not None:
            del self.processor
        self.processor = None
        self.model_config = None
        import gc

        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except (ImportError, Exception):
            pass
        console.print("[dim]llm model unloaded[/dim]")

    def is_loaded(self) -> bool:
        """check if model is currently loaded."""
        return self.model is not None

    def summarize(
        self,
        transcript_text: str,
        meeting_title: str | None = None,
        attendees: list[str] | None = None,
        has_speaker_info: bool = False,
        manual_notes_path: Path | None = None,
    ) -> str:
        """
        generate meeting summary from transcript.

        args:
            transcript_text: full transcript text
            meeting_title: optional meeting title
            attendees: optional list of attendees
            has_speaker_info: whether transcript includes speaker labels
            manual_notes_path: optional path to user's manual notes file

        returns:
            markdown-formatted summary
        """
        # load model if needed
        self._load_model()

        console.print("[cyan]generating meeting summary...[/cyan]")
        start_time = time.time()

        system_prompt = _build_system_prompt(has_speaker_info)
        user_prompt = _build_user_prompt(
            transcript_text, meeting_title, attendees, manual_notes_path
        )

        # chunk if needed
        summaries = []
        if len(user_prompt) > CHUNK_THRESHOLD:
            chunks = _chunk_transcript(transcript_text, chunk_size=CHUNK_SIZE)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task(f"processing {len(chunks)} chunks...", total=len(chunks))

                for i, chunk in enumerate(chunks):
                    chunk_prompt = f"transcript chunk {i + 1}/{len(chunks)}:\n{chunk}"
                    summary = self._generate_summary(system_prompt, chunk_prompt)
                    summaries.append(summary)
                    progress.update(task, advance=1)

            # merge summaries
            if len(summaries) > 1:
                merge_prompt = (
                    "merge these partial summaries into one final summary:\n\n"
                    + "\n---\n".join(summaries)
                )
                final_summary = self._generate_summary(system_prompt, merge_prompt)
            else:
                final_summary = summaries[0]
        else:
            # single pass for short transcripts
            with Progress(
                SpinnerColumn(),
                TextColumn("generating summary..."),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("", total=None)
                final_summary = self._generate_summary(system_prompt, user_prompt)

        duration = time.time() - start_time
        console.print(f"[green]✓[/green] summary generated in {duration:.1f}s")

        return final_summary

    def _generate_summary(self, system_prompt: str, user_prompt: str) -> str:
        """
        generate summary using the llm.

        args:
            system_prompt: system instructions
            user_prompt: user input with transcript

        returns:
            generated summary text
        """
        # build chat messages and apply template via the tokenizer
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

        if self._backend == "mlx-lm":
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(temp=self.temperature)
            raw_output = generate(
                self.model,
                self.processor,
                prompt=prompt,
                max_tokens=self.max_tokens,
                sampler=sampler,
            )
        else:
            from mlx_vlm import generate

            gen_kwargs: dict = {
                "max_tokens": self.max_tokens,
                "temp": self.temperature,
            }
            if self.enable_thinking:
                gen_kwargs["enable_thinking"] = True
                gen_kwargs["thinking_budget"] = self.thinking_budget
                gen_kwargs["thinking_start_token"] = "<think>"
                gen_kwargs["thinking_end_token"] = "</think>"

            result = generate(
                self.model,
                self.processor,
                prompt,
                **gen_kwargs,
            )
            # extract text from mlx-vlm result
            if hasattr(result, "text"):
                raw_output = result.text.strip()
            elif isinstance(result, str):
                raw_output = result.strip()
            else:
                raw_output = str(result).strip()

        raw_output = raw_output.strip()

        # always clean thinking tags — models may output them regardless of enable_thinking
        if "<think" in raw_output.lower() or "</think" in raw_output.lower():
            console.print("[dim]detected thinking tags in output, cleaning...[/dim]")
            raw_output = _clean_thinking_tags(raw_output)

        # handle untagged thinking content
        raw_output = _strip_untagged_thinking(raw_output)

        # warn if output seems empty after cleaning
        if not raw_output or len(raw_output) < 10:
            console.print("[yellow]⚠ output seems very short after cleaning[/yellow]")

        return raw_output

    # backward-compatible instance method wrappers (delegate to module-level functions)
    def _clean_thinking_tags(self, text: str) -> str:
        """remove thinking tags from llm output (delegates to module-level function)."""
        return _clean_thinking_tags(text)

    def _chunk_transcript(self, text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
        """split transcript into chunks (delegates to module-level function)."""
        return _chunk_transcript(text, chunk_size)


# ---------------------------------------------------------------------------
# Public utility functions
# ---------------------------------------------------------------------------


def save_summary(summary_text: str, base_path: Path, transcript_text: str = "") -> Path:
    """
    save summary to markdown file.

    args:
        summary_text: generated summary markdown
        base_path: base path without extension
        transcript_text: optional transcript text to append

    returns:
        path to saved summary file
    """
    summary_path = base_path.with_suffix(".summary.md")

    # ensure proper markdown formatting
    if not summary_text.startswith("## "):
        # add default structure if missing (shouldn't happen with new prompt)
        summary_text = (
            "## summary\n\n" + summary_text + "\n\n"
            "## key discussion points\n\n(none identified)\n\n"
            "## decisions\n\n(none identified)\n\n"
            "## action items\n\n(none identified)\n\n"
            "## notable quotes\n\n(none identified)"
        )

    # add metadata header
    from datetime import datetime

    header = (
        f"# Meeting Summary\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n---\n\n"
    )

    final_content = header + summary_text

    # append full transcript if provided
    if transcript_text:
        final_content += "\n\n---\n\n## Full Transcript\n\n" + transcript_text

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(final_content)

    console.print(f"[green]✓[/green] summary saved: {summary_path}")

    return summary_path


def extract_meeting_title(summary_text: str, transcript_text: str = "") -> str:
    """
    extract meeting title from summary or generate from transcript.

    args:
        summary_text: generated summary with meeting title section
        transcript_text: optional transcript for fallback title generation

    returns:
        meeting title in PascalCase format (no spaces)
    """
    # try to extract from ## Meeting Title section
    title_match = re.search(r"## Meeting Title\s*\n([^\n]+)", summary_text, re.IGNORECASE)
    if title_match:
        title = title_match.group(1).strip()
        # remove any markdown formatting
        title = re.sub(r"[*_`'\"]", "", title)
        # if already in PascalCase with no spaces, return as-is
        if " " not in title and title[0].isupper():
            return title
        # otherwise ensure PascalCase (remove spaces and capitalize)
        title = "".join(word.capitalize() for word in title.split())
        if title and len(title) > 2:
            return title

    # fallback: try to extract from first few words of summary
    summary_match = re.search(r"## Summary\s*\n([^\n]+)", summary_text, re.IGNORECASE)
    if summary_match:
        first_line = summary_match.group(1).strip()
        # extract key words (nouns/verbs)
        words = re.findall(r"\b[A-Z][a-z]+\b", first_line)
        if len(words) >= 2:
            title = "".join(words[:3])  # take first 2-3 capitalized words
            if title and len(title) > 2:
                return title

    # last resort: generate from transcript keywords
    if transcript_text:
        # find most common meaningful words
        words = re.findall(r"\b[a-z]{4,}\b", transcript_text.lower())
        if words:
            from collections import Counter

            word_counts = Counter(words)
            # filter out common words
            common_words = {
                "that",
                "this",
                "with",
                "from",
                "have",
                "been",
                "will",
                "would",
                "could",
                "should",
            }
            filtered = [(w, c) for w, c in word_counts.most_common(10) if w not in common_words]
            if filtered:
                # take top 2 words and capitalize
                title_words = [word.capitalize() for word, _ in filtered[:2]]
                return "".join(title_words) + "Meeting"

    # absolute fallback
    return "UntitledMeeting"
