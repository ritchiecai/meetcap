"""processing pipeline screen."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label, Markdown, RichLog, Static


class ProcessScreen(Screen):
    """processing pipeline screen."""

    BINDINGS = [
        Binding("escape", "back", "Back", show=True),
    ]

    def __init__(
        self,
        audio_path: Path | None = None,
        mode: str = "stt",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._audio_path = audio_path
        self._mode = mode  # "stt" = full pipeline, "summary" = summary-only
        self._processing = False
        # heartbeat state for long-running, callback-less stages
        # (e.g. mlx-whisper transcribe, summarization subprocess).
        self._heartbeat_stop: threading.Event | None = None
        self._heartbeat_thread: threading.Thread | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="process-container"):
            yield Static("Processing Pipeline", id="process-title")
            # always-visible status banner so the user knows what is
            # happening even before the first log line is written.
            yield Static(
                "Initializing...",
                id="process-status-banner",
            )
            from meetcap.tui.widgets.pipeline import PipelineProgress

            yield PipelineProgress(id="pipeline-progress")

            yield Static("Details", id="details-title")
            yield Label("Audio: --", id="process-audio-info")
            yield Label("", id="process-detail-info")

            yield Static("Live Output", id="output-title")
            yield RichLog(id="process-log", highlight=True, markup=True)

            # results section — hidden until processing completes
            yield Static("Results", id="results-title", classes="hidden")
            yield Static("Transcript", id="transcript-title", classes="hidden")
            yield Markdown("", id="result-transcript", classes="hidden")
            yield Static("Summary", id="summary-title", classes="hidden")
            yield Markdown("", id="result-summary", classes="hidden")
            yield Button(
                "Done — Back to Home",
                id="btn-done",
                variant="success",
                classes="hidden",
            )
        yield Footer()

    def on_mount(self) -> None:
        """start processing when screen mounts."""
        if self._audio_path is None:
            process_file = getattr(self.app, "process_file", None)
            if process_file:
                self._audio_path = process_file

        if self._audio_path and self._audio_path.exists():
            mode_label = "Reprocessing (summary only)" if self._mode == "summary" else "Processing"
            self._log(f"{mode_label}: {self._audio_path.name}")
            try:
                size_mb = self._audio_path.stat().st_size / (1024 * 1024)
                self.query_one("#process-audio-info", Label).update(
                    f"Audio: {self._audio_path.name} ({size_mb:.1f} MB)"
                )
            except Exception:
                pass
            self._run_pipeline()
        else:
            self._log("[yellow]No audio file to process[/yellow]")

    def _log(self, message: str) -> None:
        """add a timestamped message to the processing log."""
        timestamp = time.strftime("%H:%M:%S")
        try:
            log = self.query_one("#process-log", RichLog)
            log.write(f"[dim]{timestamp}[/dim]  {message}")
        except Exception:
            pass

    def _set_status(self, message: str, severity: str = "info") -> None:
        """update the always-visible status banner.

        severity: info (default), working, success, warn, error.
        """
        color_map = {
            "info": "cyan",
            "working": "yellow",
            "success": "green",
            "warn": "yellow",
            "error": "red",
        }
        color = color_map.get(severity, "cyan")
        try:
            banner = self.query_one("#process-status-banner", Static)
            banner.update(f"[bold {color}]{message}[/bold {color}]")
        except Exception:
            pass

    def _start_heartbeat(self, stage_label: str) -> None:
        """start a background heartbeat that prints reassurance every ~10s.

        used during opaque, long-running operations (mlx-whisper transcribe,
        summarization subprocess) where there is no progress callback.
        the message tells the user the work is still running.
        """
        # stop any previous heartbeat first
        self._stop_heartbeat()

        stop_event = threading.Event()
        self._heartbeat_stop = stop_event
        started_at = time.time()

        def _beat() -> None:
            # first reassurance after a short delay so quick stages don't
            # spam the log unnecessarily.
            interval = 10.0
            next_tick = started_at + interval
            while not stop_event.is_set():
                # poll in small increments to react to stop quickly
                if time.time() >= next_tick:
                    elapsed = time.time() - started_at
                    try:
                        self.app.call_from_thread(
                            self._log,
                            f"[dim]{stage_label}: still working... ({elapsed:.0f}s elapsed)[/dim]",
                        )
                    except Exception:
                        return
                    next_tick = time.time() + interval
                if stop_event.wait(timeout=0.5):
                    return

        thread = threading.Thread(target=_beat, daemon=True)
        self._heartbeat_thread = thread
        thread.start()

    def _stop_heartbeat(self) -> None:
        """stop the background heartbeat thread, if any."""
        if self._heartbeat_stop is not None:
            self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            try:
                self._heartbeat_thread.join(timeout=1.0)
            except Exception:
                pass
        self._heartbeat_stop = None
        self._heartbeat_thread = None

    def _check_memory(self, model_type: str, model_name: str) -> bool:
        """check memory before loading a model. returns True if safe to proceed."""
        try:
            from meetcap.utils.memory import check_memory_for_model

            sufficient, avail, needed, msg = check_memory_for_model(model_type, model_name)
            if not sufficient:
                self.app.call_from_thread(self._log, f"[red]Memory check failed: {msg}[/red]")
                self.app.call_from_thread(
                    self.notify,
                    "Not enough memory! Close other apps and try again.",
                    severity="error",
                )
                return False
            elif msg:
                self.app.call_from_thread(self._log, f"[yellow]{msg}[/yellow]")
            return True
        except Exception:
            return True  # if check fails, proceed optimistically

    @work(thread=True)
    def _run_pipeline(self) -> None:  # pragma: no cover
        """run the full processing pipeline in a background thread."""
        self._processing = True
        audio_path = self._audio_path
        if not audio_path:
            return

        try:
            from meetcap.utils.config import Config

            self.app.call_from_thread(self._set_status, "Loading configuration...", "info")

            config = Config()
            stt_engine = config.get("models", "stt_engine", "parakeet")
            llm_model = config.get(
                "models",
                "llm_model_name",
                "mlx-community/Qwen3.5-2B-OptiQ-4bit",
            )

            if self._mode == "summary":
                # summary-only mode: skip STT, read existing transcript
                self.app.call_from_thread(
                    self._update_stage, "stt", "done", detail="skipped (reprocess)"
                )
                self.app.call_from_thread(
                    self._update_stage, "diarization", "done", detail="skipped"
                )

                # find existing transcript
                transcript_text = self._read_existing_transcript(audio_path)
                if transcript_text is None:
                    self.app.call_from_thread(
                        self._set_status,
                        "No existing transcript found",
                        "error",
                    )
                    self.app.call_from_thread(
                        self._log,
                        "[red]No existing transcript found for summary-only reprocess[/red]",
                    )
                    return
                self.app.call_from_thread(
                    self._log,
                    f"Using existing transcript ({len(transcript_text)} chars)",
                )
                # no STT result object in summary-only mode
                result = None
            else:
                # full pipeline: STT
                # memory check before STT
                stt_model = self._get_stt_model_name(config, stt_engine)
                self.app.call_from_thread(
                    self._set_status,
                    "Checking available memory for STT model...",
                    "working",
                )
                if not self._check_memory("stt", stt_model):
                    self.app.call_from_thread(
                        self._set_status,
                        "Insufficient memory — close other apps",
                        "error",
                    )
                    return

                self.app.call_from_thread(self._update_stage, "stt", "active")
                stt_short = stt_model.split("/")[-1]
                self.app.call_from_thread(
                    self._set_status,
                    f"STT ({stt_engine}): loading {stt_short}...",
                    "working",
                )
                self.app.call_from_thread(
                    self._log,
                    f"Starting STT ({stt_engine}). Loading {stt_short} — first run may download model (~1-3 GB).",
                )
                stt_start = time.time()
                # heartbeat reassures during opaque transcribe call
                self.app.call_from_thread(self._start_heartbeat, "STT")

                result = self._run_stt(audio_path, config, stt_engine)

                self.app.call_from_thread(self._stop_heartbeat)
                stt_time = time.time() - stt_start
                seg_count = len(result.segments) if result else 0
                self.app.call_from_thread(
                    self._update_stage,
                    "stt",
                    "done",
                    timing=stt_time,
                    detail=f"{seg_count} segments",
                )

                if not result:
                    self.app.call_from_thread(
                        self._set_status,
                        "STT produced no result — check audio language / engine",
                        "error",
                    )
                    self.app.call_from_thread(
                        self._log,
                        "[red]STT produced no result[/red]",
                    )
                    return

                if seg_count == 0:
                    self.app.call_from_thread(
                        self._set_status,
                        "STT returned 0 segments — audio may be silent or unsupported language",
                        "warn",
                    )
                    self.app.call_from_thread(
                        self._log,
                        "[yellow]Warning: STT returned 0 segments. "
                        "If the audio contains speech, try switching engine "
                        "(e.g. mlx-whisper for Chinese/Japanese).[/yellow]",
                    )

                self.app.call_from_thread(
                    self._set_status,
                    f"STT done in {stt_time:.1f}s — {seg_count} segments",
                    "success",
                )
                transcript_text = " ".join(seg.text for seg in result.segments)

                # stage 2: diarization
                enable_diar = config.get("models", "enable_speaker_diarization", True)
                diar_backend = config.get("models", "diarization_backend", "sherpa")
                if enable_diar and diar_backend == "sherpa" and stt_engine != "vosk":
                    self.app.call_from_thread(self._update_stage, "diarization", "active")
                    self.app.call_from_thread(
                        self._set_status,
                        "Diarization: identifying speakers...",
                        "working",
                    )
                    self.app.call_from_thread(self._log, "Running diarization...")
                    diar_start = time.time()
                    self.app.call_from_thread(self._start_heartbeat, "Diarization")
                    self._run_diarization(audio_path, config, result)
                    self.app.call_from_thread(self._stop_heartbeat)
                    diar_time = time.time() - diar_start
                    self.app.call_from_thread(
                        self._update_stage,
                        "diarization",
                        "done",
                        timing=diar_time,
                    )
                    self.app.call_from_thread(
                        self._set_status,
                        f"Diarization done in {diar_time:.1f}s",
                        "success",
                    )
                else:
                    self.app.call_from_thread(
                        self._update_stage,
                        "diarization",
                        "done",
                        detail="skipped",
                    )

            # memory check before LLM (skip for oMLX — it manages its own memory)
            llm_backend = config.get("llm", "backend", "mlx-lm")  # type: ignore[union-attr]
            if llm_backend != "omlx":
                self.app.call_from_thread(
                    self._set_status,
                    "Checking available memory for LLM...",
                    "working",
                )
                if not self._check_memory("llm", llm_model):
                    self.app.call_from_thread(
                        self._set_status,
                        "Insufficient memory for LLM — close other apps",
                        "error",
                    )
                    return

            # stage 3: summarization
            self.app.call_from_thread(self._update_stage, "summarization", "active")
            model_short = llm_model.split("/")[-1]
            backend_label = "oMLX" if llm_backend == "omlx" else "mlx-lm"
            self.app.call_from_thread(
                self._set_status,
                f"Summarizing with {model_short} via {backend_label} "
                "(this can take 30s-3min, hang tight)...",
                "working",
            )
            self.app.call_from_thread(
                self._log,
                f"Summarizing with {model_short} via {backend_label}. "
                f"Transcript size: {len(transcript_text)} chars. "
                "Long meetings will be auto-chunked.",
            )
            sum_start = time.time()
            self.app.call_from_thread(self._start_heartbeat, "Summarization")
            summary = self._run_summarization(transcript_text, config)
            self.app.call_from_thread(self._stop_heartbeat)
            sum_time = time.time() - sum_start
            self.app.call_from_thread(
                self._log, f"Summary generated in {sum_time:.1f}s ({backend_label})"
            )
            self.app.call_from_thread(self._update_stage, "summarization", "done", timing=sum_time)
            self.app.call_from_thread(
                self._set_status,
                f"Summary done in {sum_time:.1f}s",
                "success",
            )

            # stage 4: organize
            self.app.call_from_thread(self._update_stage, "organize", "active")
            self.app.call_from_thread(self._set_status, "Organizing files...", "working")
            self.app.call_from_thread(self._log, "Organizing files...")
            self._organize_files(audio_path, result, summary, transcript_text)
            self.app.call_from_thread(self._update_stage, "organize", "done")

            self.app.call_from_thread(self._set_status, "All done!", "success")
            self.app.call_from_thread(self._log, "[green]Processing complete![/green]")
            self.app.call_from_thread(self.notify, "Processing complete!", severity="information")
            self.app.call_from_thread(self._show_results, transcript_text, summary)
            self.app.call_from_thread(self._show_done_button)

        except Exception as e:
            self.app.call_from_thread(self._stop_heartbeat)
            self.app.call_from_thread(self._set_status, f"Error: {e}", "error")
            self.app.call_from_thread(self._log, f"[red]Error: {e}[/red]")
            self.app.call_from_thread(self._show_done_button)
        finally:
            self._stop_heartbeat()
            self._processing = False

    @staticmethod
    def _get_stt_model_name(config: object, stt_engine: str) -> str:
        """return the configured model name for a given STT engine."""
        if stt_engine in ("parakeet", "parakeet-tdt"):
            return config.get(  # type: ignore[union-attr]
                "models", "parakeet_model_name", "mlx-community/parakeet-tdt-0.6b-v3"
            )
        elif stt_engine in ("mlx-whisper", "mlx"):
            return config.get(  # type: ignore[union-attr]
                "models", "mlx_stt_model_name", "mlx-community/whisper-large-v3-turbo"
            )
        elif stt_engine in ("faster-whisper", "fwhisper"):
            return config.get(  # type: ignore[union-attr]
                "models", "stt_model_name", "large-v3"
            )
        return "whisper.cpp"

    def _read_existing_transcript(self, audio_path: Path) -> str | None:
        """read existing transcript text file for summary-only reprocessing."""
        # try the standard naming convention
        base = audio_path.with_suffix("")
        txt_path = base.with_suffix(".transcript.txt")
        if txt_path.exists():
            try:
                return txt_path.read_text(encoding="utf-8")
            except Exception:
                pass

        # try finding any transcript file in the same directory
        for f in audio_path.parent.glob("*.transcript.txt"):
            try:
                return f.read_text(encoding="utf-8")
            except Exception:
                continue

        return None

    def _run_stt(self, audio_path: Path, config: object, stt_engine: str) -> object | None:
        """run speech-to-text transcription.

        mirrors the service construction logic in cli.py _process_audio_to_transcript.
        """
        try:
            # map config names to CLI names (same as cli.py)
            engine = stt_engine
            if engine == "faster-whisper":
                engine = "fwhisper"
            elif engine == "mlx-whisper":
                engine = "mlx"

            if engine == "parakeet":
                from meetcap.services.transcription import ParakeetService

                model = config.get(  # type: ignore[union-attr]
                    "models",
                    "parakeet_model_name",
                    "mlx-community/parakeet-tdt-0.6b-v3",
                )
                service = ParakeetService(model_name=model)
            elif engine == "mlx":
                from meetcap.services.transcription import MlxWhisperService

                model_name = config.get(  # type: ignore[union-attr]
                    "models",
                    "mlx_stt_model_name",
                    "mlx-community/whisper-large-v3-turbo",
                )
                service = MlxWhisperService(model_name=model_name, auto_download=True)
            elif engine == "fwhisper":
                from meetcap.services.transcription import FasterWhisperService

                model_name = config.get(  # type: ignore[union-attr]
                    "models", "stt_model_name", "large-v3"
                )
                model_path = config.expand_path(  # type: ignore[union-attr]
                    config.get("models", "stt_model_path")  # type: ignore[union-attr]
                )
                service = FasterWhisperService(
                    model_path=str(model_path),
                    model_name=model_name,
                    auto_download=True,
                )
            elif engine == "vosk":
                from meetcap.services.transcription import (
                    VoskTranscriptionService,
                )

                model_path = config.expand_path(  # type: ignore[union-attr]
                    config.get("models", "vosk_model_path")  # type: ignore[union-attr]
                )
                service = VoskTranscriptionService(model_path=str(model_path))
            else:
                from meetcap.services.transcription import WhisperCppService

                model_path = config.get(  # type: ignore[union-attr]
                    "models", "stt_model_path"
                )
                service = WhisperCppService(model_path=model_path)

            service.load_model()
            self.app.call_from_thread(
                self._set_status,
                f"STT ({stt_engine}): model loaded, transcribing audio...",
                "working",
            )
            self.app.call_from_thread(
                self._log,
                "[green]Model loaded.[/green] Transcribing audio "
                "(no progress bar — this is normal, watch the spinner)...",
            )
            result = service.transcribe(audio_path)
            service.unload_model()
            return result
        except Exception as e:
            self.app.call_from_thread(self._log, f"[red]STT error: {e}[/red]")
            return None

    def _run_diarization(self, audio_path: Path, config: object, result: object) -> None:
        """run speaker diarization via sherpa-onnx.

        mirrors the diarization logic in cli.py _process_audio_to_transcript.
        """
        try:
            from meetcap.services.diarization import (
                SherpaOnnxDiarizationService,
                assign_speakers,
            )

            models_dir = config.expand_path(  # type: ignore[union-attr]
                config.get(  # type: ignore[union-attr]
                    "paths", "models_dir", "~/.meetcap/models"
                )
            )
            seg_model = str(models_dir / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx")
            emb_model = str(
                models_dir / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
            )
            num_speakers = int(
                config.get(  # type: ignore[union-attr]
                    "models", "sherpa_num_speakers", -1
                )
            )
            threshold = float(
                config.get(  # type: ignore[union-attr]
                    "models", "sherpa_cluster_threshold", 0.90
                )
            )
            provider = str(
                config.get(  # type: ignore[union-attr]
                    "models", "sherpa_provider", "cpu"
                )
            )
            num_threads = int(
                config.get(  # type: ignore[union-attr]
                    "models", "sherpa_num_threads", 4
                )
            )

            diar_service = SherpaOnnxDiarizationService(
                segmentation_model=seg_model,
                embedding_model=emb_model,
                num_speakers=num_speakers,
                threshold=threshold,
                provider=provider,
                num_threads=num_threads,
            )
            diar_segments = diar_service.diarize(audio_path)
            result.segments, result.speakers = assign_speakers(  # type: ignore[union-attr]
                result.segments,
                diar_segments,  # type: ignore[union-attr]
            )
            result.diarization_enabled = True  # type: ignore[union-attr]
            diar_service.unload_model()
        except ImportError:
            self.app.call_from_thread(
                self._log,
                "[yellow]sherpa-onnx not installed, skipping diarization[/yellow]",
            )
        except FileNotFoundError as e:
            self.app.call_from_thread(
                self._log, f"[yellow]Diarization models not found: {e}[/yellow]"
            )
        except Exception as e:
            self.app.call_from_thread(self._log, f"[yellow]Diarization skipped: {e}[/yellow]")

    def _run_summarization(  # pragma: no cover
        self, transcript: str, config: object
    ) -> str:
        """run LLM summarization in a subprocess to avoid fd conflicts with textual."""
        import json
        import subprocess
        import sys
        import tempfile

        model_name = config.get(  # type: ignore[union-attr]
            "models",
            "llm_model_name",
            "mlx-community/Qwen3.5-2B-OptiQ-4bit",
        )
        temperature = float(
            config.get("llm", "temperature", 0.4)  # type: ignore[union-attr]
        )
        max_tokens = int(
            config.get("llm", "max_tokens", 4096)  # type: ignore[union-attr]
        )
        enable_thinking = bool(
            config.get("llm", "enable_thinking", False)  # type: ignore[union-attr]
        )
        thinking_budget = int(
            config.get("llm", "thinking_budget", 512)  # type: ignore[union-attr]
        )
        llm_backend = config.get("llm", "backend", "mlx-lm")  # type: ignore[union-attr]

        try:
            # write transcript to temp file (avoid shell escaping issues)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as f:
                f.write(transcript)
                transcript_path = f.name

            if llm_backend == "omlx":
                # oMLX: call via OpenAI-compatible API (no fd issues)
                omlx_url = config.get("llm", "omlx_base_url", "http://localhost:8000/v1")  # type: ignore[union-attr]
                omlx_key = config.get("llm", "omlx_api_key", "")  # type: ignore[union-attr]
                omlx_timeout = int(config.get("llm", "omlx_timeout", 300))  # type: ignore[union-attr]

                script = f"""
import json, sys
from meetcap.services.summarization import OmlxSummarizationService
with open({transcript_path!r}, encoding="utf-8") as f:
    transcript = f.read()
service = OmlxSummarizationService(
    model_name={model_name!r},
    base_url={omlx_url!r},
    temperature={temperature},
    max_tokens={max_tokens},
    enable_thinking={enable_thinking},
    thinking_budget={thinking_budget},
    api_key={omlx_key!r},
    timeout={omlx_timeout},
)
result = service.summarize(transcript)
print(json.dumps({{"summary": result}}))
"""
            else:
                # mlx-lm: run in isolated subprocess to avoid
                # "bad value(s) in fds_to_keep" error from mlx-vlm inside textual
                script = f"""
import json, sys
from meetcap.services.summarization import SummarizationService
with open({transcript_path!r}, encoding="utf-8") as f:
    transcript = f.read()
service = SummarizationService(
    model_name={model_name!r},
    temperature={temperature},
    max_tokens={max_tokens},
    enable_thinking={enable_thinking},
    thinking_budget={thinking_budget},
)
service.load_model()
result = service.summarize(transcript)
service.unload_model()
print(json.dumps({{"summary": result}}))
"""
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=600 if llm_backend == "omlx" else 300,
            )

            # clean up temp file
            from pathlib import Path

            Path(transcript_path).unlink(missing_ok=True)

            if proc.returncode == 0:
                # parse the last line of stdout as JSON
                for line in reversed(proc.stdout.strip().split("\n")):
                    line = line.strip()
                    if line.startswith("{"):
                        data = json.loads(line)
                        return data.get("summary", "")
            else:
                stderr = proc.stderr.strip()[-200:] if proc.stderr else "unknown"
                self.app.call_from_thread(
                    self._log,
                    f"[red]Summarization failed: {stderr}[/red]",
                )
                return ""
        except Exception as e:
            self.app.call_from_thread(self._log, f"[red]Summarization error: {e}[/red]")
        return ""

    def _organize_files(  # pragma: no cover
        self,
        audio_path: Path,
        result: object,
        summary: str,
        transcript_text: str,
    ) -> None:
        """save transcript and summary files."""
        try:
            from meetcap.services.summarization import save_summary
            from meetcap.services.transcription import save_transcript

            base_path = audio_path.with_suffix("")
            if result is not None:
                save_transcript(result, base_path)
            if summary:
                save_summary(summary, base_path, transcript_text)
        except Exception as e:
            self.app.call_from_thread(self._log, f"[yellow]File save warning: {e}[/yellow]")

    def _update_stage(
        self,
        name: str,
        status: str,
        timing: float = 0.0,
        detail: str = "",
    ) -> None:
        """update a pipeline stage's display."""
        progress_map = {
            "pending": 0,
            "active": 50,
            "done": 100,
            "error": 0,
        }
        try:
            from meetcap.tui.widgets.pipeline import PipelineProgress

            pipeline = self.query_one("#pipeline-progress", PipelineProgress)
            pipeline.update_stage(
                name,
                status,
                progress=progress_map.get(status, 0),
                timing=timing,
                detail=detail,
            )
        except Exception:
            pass

    def _show_results(self, transcript: str, summary: str) -> None:
        """show transcript and summary after processing."""
        try:
            self.query_one("#results-title").remove_class("hidden")

            # show transcript
            self.query_one("#transcript-title").remove_class("hidden")
            transcript_md = self.query_one("#result-transcript", Markdown)
            transcript_md.remove_class("hidden")
            # format transcript as markdown code block for readability
            display_text = transcript[:3000]
            if len(transcript) > 3000:
                display_text += "\n\n*... (truncated)*"
            transcript_md.update(display_text or "*No transcript generated*")

            # show summary
            self.query_one("#summary-title").remove_class("hidden")
            summary_md = self.query_one("#result-summary", Markdown)
            summary_md.remove_class("hidden")
            summary_md.update(summary or "*No summary generated*")
        except Exception:
            pass

    def _show_done_button(self) -> None:
        """show the done button after processing finishes."""
        try:
            self.query_one("#btn-done", Button).remove_class("hidden")
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """handle button presses."""
        if event.button.id == "btn-done":
            self.app.pop_screen()

    def action_back(self) -> None:
        """handle escape to go back."""
        if not self._processing:
            self._stop_heartbeat()
            self.app.pop_screen()
        else:
            self.notify("Processing in progress...", severity="warning")
