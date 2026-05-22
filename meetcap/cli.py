"""command-line interface for meetcap"""

import os
import queue
import signal
import sys
import threading
import time
import urllib.request
from pathlib import Path

import typer
from rich import prompt
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from meetcap import __version__
from meetcap.core.devices import (
    find_device_by_index,
    find_device_by_name,
    list_audio_devices,
    print_devices,
    select_best_device,
)
from meetcap.core.hotkeys import HotkeyManager, PermissionChecker
from meetcap.core.recorder import AudioRecorder
from meetcap.services.model_download import (
    ensure_mlx_llm_model,
    ensure_mlx_whisper_model,
    ensure_vosk_model,
    ensure_vosk_spk_model,
    ensure_whisper_model,
    verify_mlx_llm_model,
    verify_mlx_whisper_model,
    verify_vosk_model,
    verify_whisper_model,
)
from meetcap.services.summarization import (
    OmlxSummarizationService,
    SummarizationService,
    extract_meeting_title,
    save_summary,
)
from meetcap.services.transcription import (
    FasterWhisperService,
    MlxWhisperService,
    ParakeetService,
    VoskTranscriptionService,
    WhisperCppService,
    save_transcript,
)
from meetcap.utils.config import Config
from meetcap.utils.logger import ErrorHandler, logger
from meetcap.utils.memory import (
    MemoryMonitor,
    check_memory_for_model,
    check_memory_pressure,
    preflight_memory_check,
)

console = Console()
app = typer.Typer(
    name="meetcap",
    help="offline meeting recorder & summarizer for macos",
    add_completion=False,
)


def create_notes_file(config: Config, recording_dir: Path) -> Path | None:
    """Create notes.md file in recording directory."""
    notes_path = recording_dir / "notes.md"
    try:
        template = config.get("notes", "template", "# Meeting Notes\n\n*Add your notes here*\n")
        with open(notes_path, "w", encoding="utf-8") as f:
            f.write(template)
        return notes_path
    except Exception as e:
        console.print(f"[yellow]⚠[/yellow] could not create notes file: {e}")
        return None


def read_manual_notes(notes_path: Path) -> str:
    """Read manual notes file with error handling."""
    if not notes_path.exists():
        return ""

    try:
        with open(notes_path, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        console.print(f"[yellow]⚠[/yellow] could not read manual notes: {e}")
        return ""


def validate_auto_stop_time(minutes: int) -> bool:
    """validate that auto stop time is one of the supported options."""
    return minutes in [0, 30, 60, 90, 120]


class BackupManager:
    """manages file backups for reprocessing operations"""

    def __init__(self):
        """initialize backup manager."""
        self.backups = []

    def create_backup(self, file_path: Path) -> Path | None:
        """
        create backup with .backup extension.

        args:
            file_path: path to file to backup

        returns:
            path to backup file or None if file doesn't exist
        """
        if not file_path.exists():
            return None

        # backup notes files specifically
        if file_path.name == "notes.md":
            backup_path = file_path.with_suffix(file_path.suffix + ".backup")
            try:
                import shutil

                shutil.copy2(file_path, backup_path)
                self.backups.append(backup_path)
                return backup_path
            except Exception as e:
                logger.error(f"failed to create backup for {file_path}: {e}")
                return None

        backup_path = file_path.with_suffix(file_path.suffix + ".backup")
        try:
            import shutil

            shutil.copy2(file_path, backup_path)
            self.backups.append(backup_path)
            return backup_path
        except Exception as e:
            logger.error(f"failed to create backup for {file_path}: {e}")
            return None

    def restore_backup(self, file_path: Path) -> bool:
        """
        restore from backup if exists.

        args:
            file_path: original file path to restore

        returns:
            True if restored, False otherwise
        """
        backup_path = file_path.with_suffix(file_path.suffix + ".backup")
        if backup_path.exists():
            try:
                import shutil

                shutil.move(str(backup_path), str(file_path))
                if backup_path in self.backups:
                    self.backups.remove(backup_path)
                return True
            except Exception as e:
                logger.error(f"failed to restore backup for {file_path}: {e}")
                return False
        return False

    def cleanup_backups(self, directory: Path) -> None:
        """
        remove all .backup files after success.

        args:
            directory: directory to clean up
        """
        for backup_file in directory.glob("*.backup"):
            try:
                backup_file.unlink()
            except Exception as e:
                logger.warning(f"failed to remove backup {backup_file}: {e}")
        self.backups.clear()

    def restore_all(self) -> None:
        """restore all tracked backups."""
        for backup_path in list(self.backups):  # copy list to avoid modification during iteration
            # remove .backup suffix to get original path
            backup_str = str(backup_path)
            if backup_str.endswith(".backup"):
                original_path = Path(backup_str[:-7])  # remove last 7 chars (".backup")
                self.restore_backup(original_path)


class RecordingOrchestrator:
    """orchestrates the recording, transcription, and summarization workflow"""

    def __init__(self, config: Config):
        """initialize orchestrator with config."""
        self.config = config
        self.recorder = None
        self.hotkey_manager = None
        self.stop_event = threading.Event()
        self.interrupt_count = 0
        self.last_interrupt_time = 0
        self.processing_complete = False
        self.graceful_stop_requested = False
        self.memory_monitor = None
        self.enable_memory_monitoring = config.get("memory", "enable_monitoring", False)
        self.auto_stop_minutes = None
        self.auto_stop_timer = None
        self.auto_stop_start_time = None

        # Timer control attributes
        self.timer_lock = threading.Lock()
        self.timer_operations_queue = queue.Queue()

    def _cleanup_service(self, service: object, service_name: str) -> None:
        """unload model and run cleanup for a service."""
        if hasattr(service, "unload_model"):
            try:
                service.unload_model()
            except Exception as e:
                logger.warning(f"failed to cleanup {service_name}: {e}")
        if self.config.get("memory", "aggressive_gc", True):
            import gc

            gc.collect()

    def run(
        self,
        device: str | None = None,
        output_dir: str | None = None,
        sample_rate: int | None = None,
        channels: int | None = None,
        stt_engine: str | None = None,
        llm_model: str | None = None,
        seed: int | None = None,
        auto_stop: int | None = None,
        audio_format: str | None = None,
        opus_bitrate: int | None = None,
        flac_compression: int | None = None,
    ) -> None:
        """
        run the complete recording workflow.

        args:
            device: device name or index
            output_dir: output directory path
            sample_rate: audio sample rate
            channels: number of channels
            stt_engine: stt engine to use
            llm_model: huggingface repo id for llm model
            seed: random seed for llm
            auto_stop: minutes after which to automatically stop recording
            audio_format: recording format (wav/opus/flac)
            opus_bitrate: opus bitrate in kbps
            flac_compression: flac compression level (0-8)
        """
        # Store auto_stop_minutes for timer functionality
        self.auto_stop_minutes = auto_stop

        # setup configuration
        if output_dir:
            out_path = Path(output_dir)
        else:
            out_path = self.config.expand_path(self.config.get("paths", "out_dir"))

        sample_rate = sample_rate or self.config.get("audio", "sample_rate")
        channels = channels or self.config.get("audio", "channels")

        # Get audio format from config if not specified
        if audio_format is None:
            audio_format = self.config.get("audio", "format", "wav")
        if opus_bitrate is None:
            opus_bitrate = self.config.get("audio", "opus_bitrate", 32)
        if flac_compression is None:
            flac_compression = self.config.get("audio", "flac_compression_level", 5)

        # Convert string format to AudioFormat enum
        from meetcap.utils.config import AudioFormat

        try:
            format_enum = AudioFormat(audio_format.lower())
        except ValueError:
            console.print(f"[yellow]warning: invalid format '{audio_format}', using WAV[/yellow]")
            format_enum = AudioFormat.WAV

        # initialize recorder
        self.recorder = AudioRecorder(
            output_dir=out_path,
            sample_rate=sample_rate,
            channels=channels,
        )

        # find audio device
        devices = list_audio_devices()
        if not devices:
            ErrorHandler.handle_runtime_error(RuntimeError("no audio input devices found"))
            return  # this line shouldn't be reached due to sys.exit, but helps with testing

        selected_device = None
        if device:
            # try as index first
            try:
                device_index = int(device)
                selected_device = find_device_by_index(devices, device_index)
            except ValueError:
                # try as name
                selected_device = find_device_by_name(devices, device)
        else:
            # auto-select best device
            preferred = self.config.get("audio", "preferred_device")
            selected_device = find_device_by_name(devices, preferred)
            if not selected_device:
                selected_device = select_best_device(devices)

        if not selected_device:
            ErrorHandler.handle_config_error(ValueError(f"device not found: {device}"))
            return  # this line shouldn't be reached due to sys.exit, but helps with testing

        # show recording banner
        console.print(
            Panel(
                f"[bold cyan]meetcap v{__version__}[/bold cyan]\n"
                f"[green]starting recording...[/green]",
                title="🎙️ meeting recorder",
                expand=False,
            )
        )

        # setup hotkey handler with timer support
        prefix_key = self.config.get("hotkey", "prefix", "<ctrl>+a")
        self.hotkey_manager = HotkeyManager(self._stop_recording, self._timer_callback, prefix_key)
        hotkey_combo = self.config.get("hotkey", "stop")

        # setup signal handlers for Ctrl-C and SIGTERM
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_terminate)

        try:
            # start recording
            self.recorder.start_recording(
                device_index=selected_device.index,
                device_name=selected_device.name,
                audio_format=format_enum,
                opus_bitrate=opus_bitrate,
                flac_compression=flac_compression,
            )

            # start hotkey listener
            self.hotkey_manager.start(hotkey_combo)
            console.print("[cyan]⌃C[/cyan] press once to stop recording, twice to exit")

            # If auto_stop is specified, start timer thread
            if self.auto_stop_minutes is not None and self.auto_stop_minutes > 0:
                self.auto_stop_start_time = time.time()  # Track our own start time
                self._start_auto_stop_timer()

            # show progress while recording
            self._show_recording_progress()

            # stop recording (triggered by hotkey, Ctrl-C, or auto timer)
            final_path = self.recorder.stop_recording()
            if not final_path:
                ErrorHandler.handle_runtime_error(RuntimeError("recording failed or was empty"))

            # cleanup auto stop timer
            if self.auto_stop_timer and self.auto_stop_timer.is_alive():
                self.auto_stop_timer.join(timeout=1.0)

            # prompt user for processing confirmation
            if console.is_interactive:
                proceed = prompt.Confirm.ask(
                    "Proceed with transcription and summarization?", default=True
                )
            else:
                proceed = True

            if proceed:
                # run transcription and summarization
                self.processing_complete = False
                self._process_recording(
                    audio_path=final_path,
                    stt_engine=stt_engine,
                    llm_model=llm_model,
                    seed=seed,
                )
                self.processing_complete = True
            else:
                self.processing_complete = True
                console.print(f"Processing skipped. Audio file saved at: {final_path}")

        except KeyboardInterrupt:
            # handle KeyboardInterrupt based on current state
            if self.graceful_stop_requested:
                # this is a second Ctrl-C after we already handled a graceful stop
                console.print("\n[red]force exit requested[/red]")
                if self.recorder and self.recorder.is_recording():
                    self.recorder.stop_recording()
                return
            elif self.recorder and self.recorder.is_recording():
                # if still recording, this means it's the first Ctrl-C during recording
                console.print("\n[yellow]⏹[/yellow] stopping recording...")
                final_path = self.recorder.stop_recording()
                if final_path:
                    # prompt user for processing confirmation
                    if console.is_interactive:
                        proceed = prompt.Confirm.ask(
                            "Proceed with transcription and summarization?", default=True
                        )
                    else:
                        proceed = True

                    if proceed:
                        # continue with processing
                        self.processing_complete = False
                        self._process_recording(
                            audio_path=final_path,
                            stt_engine=stt_engine,
                            llm_model=llm_model,
                            seed=seed,
                        )
                        self.processing_complete = True
                    else:
                        self.processing_complete = True
                        console.print(f"Processing skipped. Audio file saved at: {final_path}")
                else:
                    console.print(
                        "\n[yellow]operation cancelled - no recording to process[/yellow]"
                    )
            else:
                # if not recording, this is during processing or already stopped
                console.print("\n[yellow]operation cancelled[/yellow]")
        except Exception as e:
            ErrorHandler.handle_runtime_error(e)
        finally:
            # CRITICAL: clean up ffmpeg subprocess before anything else
            if self.recorder:
                self.recorder.cleanup()
            if self.hotkey_manager:
                self.hotkey_manager.stop()
            # restore default signal handlers
            signal.signal(signal.SIGINT, signal.default_int_handler)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)

    def _handle_interrupt(self, signum, frame) -> None:
        """handle Ctrl-C interrupt signal."""
        current_time = time.time()

        # check for double Ctrl-C (within 2 seconds)
        if current_time - self.last_interrupt_time < 2.0:
            self.interrupt_count += 1
        else:
            self.interrupt_count = 1

        self.last_interrupt_time = current_time

        if self.interrupt_count >= 2:
            # double Ctrl-C: exit immediately
            console.print("\n[red]double interrupt - exiting immediately[/red]")
            if self.recorder and self.recorder.is_recording():
                self.recorder.stop_recording()
            sys.exit(1)
        else:
            # single Ctrl-C: stop recording gracefully
            if self.recorder and self.recorder.is_recording():
                console.print(
                    "\n[yellow]⏹[/yellow] stopping recording (press Ctrl-C again to force exit)"
                )
                self._stop_recording()
                self.graceful_stop_requested = True
                # don't let KeyboardInterrupt propagate - we want to continue to processing
                return
            elif not self.processing_complete:
                console.print(
                    "\n[yellow]processing in progress (press Ctrl-C again to force exit)[/yellow]"
                )
                # don't exit during processing, let it continue
                return
            else:
                # if not recording and processing is done, exit
                sys.exit(0)

    def _handle_terminate(self, signum, frame) -> None:
        """handle SIGTERM for graceful shutdown."""
        logger.info("received SIGTERM, shutting down")
        if self.recorder:
            self.recorder.cleanup()
        sys.exit(0)

    def _stop_recording(self) -> None:
        """callback for hotkey to stop recording."""
        self.stop_event.set()
        # Signal to the progress display thread to stop
        if self.recorder:
            self.recorder._stop_event.set()

    def extend_timer(self, minutes: int) -> None:
        """Extend current timer by specified minutes."""
        self.timer_operations_queue.put(("extend", minutes))

    def cancel_timer(self) -> None:
        """Cancel the current auto-stop timer."""
        self.timer_operations_queue.put(("cancel", None))

    def set_new_timer(self, minutes: int) -> None:
        """Set new timer duration from current time."""
        self.timer_operations_queue.put(("set", minutes))

    def get_timer_status(self) -> dict:
        """Get current timer status information."""
        with self.timer_lock:
            if not self.auto_stop_minutes or not self.auto_stop_start_time:
                return {"active": False}

            elapsed = time.time() - self.auto_stop_start_time
            total_seconds = self.auto_stop_minutes * 60
            remaining = max(0, total_seconds - elapsed)

            return {
                "active": True,
                "duration_minutes": self.auto_stop_minutes,
                "elapsed_seconds": elapsed,
                "remaining_seconds": remaining,
                "start_time": self.auto_stop_start_time,
            }

    def _timer_callback(self, action: str, value: int = None) -> None:
        """Handle timer-related hotkey callbacks."""
        try:
            if action == "extend":
                minutes = value or 30  # Default to 30 minutes
                self.extend_timer(minutes)
            elif action == "cancel":
                self.cancel_timer()
            elif action == "menu":
                self._show_timer_status()
            elif action == "set":
                minutes = value or 60  # Default to 60 minutes
                self.set_new_timer(minutes)
        except Exception:
            # Silent error handling - errors don't disrupt progress display
            pass

    def _show_timer_status(self) -> None:
        """Display current timer status in a less disruptive way."""
        status = self.get_timer_status()
        if not status["active"]:
            # Just flash a brief status without disrupting the progress line too much
            console.print("\r[dim]No timer active[/dim]", end="")
            # Give user a moment to see it, then let progress resume
            import threading

            threading.Timer(1.0, lambda: console.print("\r", end="")).start()
        else:
            remaining_mins = int(status["remaining_seconds"] // 60)
            remaining_secs = int(status["remaining_seconds"] % 60)
            # Show brief status inline, then let progress resume
            console.print(
                f"\r[dim]Timer: {status['duration_minutes']}min, {remaining_mins:02d}:{remaining_secs:02d} left[/dim]",
                end="",
            )
            # Give user a moment to see it, then let progress resume
            import threading

            threading.Timer(2.0, lambda: console.print("\r", end="")).start()

    def _process_timer_operation(self) -> None:
        """Process queued timer operations safely."""
        try:
            operation, value = self.timer_operations_queue.get_nowait()

            with self.timer_lock:
                if operation == "extend":
                    if self.auto_stop_minutes and self.auto_stop_start_time:
                        self.auto_stop_minutes += value
                        # Silent operation - timer extension reflected in progress display
                    else:
                        # Silent - no active timer to extend
                        pass

                elif operation == "cancel":
                    if self.auto_stop_minutes:
                        # Silent operation - timer cancellation reflected in progress display
                        self.auto_stop_minutes = None
                        self.auto_stop_start_time = None
                    else:
                        # Silent - no active timer to cancel
                        pass

                elif operation == "set":
                    self.auto_stop_minutes = value
                    self.auto_stop_start_time = time.time()
                    # Silent operation - timer setting reflected in progress display

        except queue.Empty:
            pass  # No operations to process
        except Exception:
            # Silent error handling - errors don't disrupt progress display
            pass

    def _start_auto_stop_timer(self) -> None:
        """start background timer for automatic stopping."""
        self.auto_stop_timer = threading.Thread(target=self._auto_stop_worker, daemon=True)
        self.auto_stop_timer.start()

    def _auto_stop_worker(self) -> None:
        """Enhanced background worker that monitors recording time and processes timer operations."""
        import time

        # Continue running even if no initial timer - operations may add one
        while not self.stop_event.is_set():
            # Process any pending timer operations
            if not self.timer_operations_queue.empty():
                self._process_timer_operation()

            # Check timer status with lock for thread safety
            with self.timer_lock:
                if not self.auto_stop_minutes or not self.auto_stop_start_time:
                    time.sleep(1)  # No active timer, just wait
                    continue

                elapsed = time.time() - self.auto_stop_start_time
                stop_seconds = self.auto_stop_minutes * 60

                if elapsed >= stop_seconds:
                    console.print(
                        f"\n[yellow]⏱️[/yellow] automatically stopping recording after {self.auto_stop_minutes} minutes"
                    )
                    self._stop_recording()
                    break

            time.sleep(1)  # Check every second

    def _show_recording_progress(self) -> None:
        """display recording progress until stopped."""
        start_time = time.time()
        hotkey_str = (
            self.config.get("hotkey", "stop")
            .replace("<cmd>", "⌘")
            .replace("<shift>", "⇧")
            .replace("+", "")
            .upper()
        )

        try:
            while not self.stop_event.is_set():
                elapsed = time.time() - start_time
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)

                # Build progress display string
                progress_str = f"[cyan]recording[/cyan] {minutes:02d}:{seconds:02d}"

                # Add notes file path display
                if self.recorder and self.recorder.session:
                    recording_dir = self.recorder.session.output_path.parent
                    notes_path = recording_dir / "notes.md"
                    if notes_path.exists():
                        progress_str += f" [dim]notes: {notes_path.absolute()}[/dim]"

                # Add time remaining and shortcuts if auto-stop is active
                if (
                    self.auto_stop_minutes
                    and self.auto_stop_minutes > 0
                    and self.auto_stop_start_time
                ):
                    # Use our independent timer for consistency
                    auto_elapsed = time.time() - self.auto_stop_start_time
                    total_seconds = self.auto_stop_minutes * 60
                    remaining_seconds = max(0, total_seconds - auto_elapsed)
                    remaining_minutes = int(remaining_seconds // 60)
                    remaining_seconds = int(remaining_seconds % 60)
                    progress_str += f" [dim](⏱️ auto-stop in {remaining_minutes:02d}:{remaining_seconds:02d})[/dim]"
                    # Add prefix-based timer shortcuts when timer is active
                    prefix_display = (
                        self.config.get("hotkey", "prefix", "<ctrl>+a")
                        .replace("<ctrl>", "⌃")
                        .replace("<cmd>", "⌘")
                        .replace("<alt>", "⌥")
                        .replace("<shift>", "⇧")
                        .replace("+", "")
                        .upper()
                    )
                    progress_str += (
                        f" [dim]({prefix_display} then c=cancel e=extend t=menu 1/2/3=quick)[/dim]"
                    )
                else:
                    progress_str += f" [dim]({hotkey_str} or ⌃C to stop)[/dim]"

                console.print(progress_str, end="\r")

                # use stop_event.wait() instead of time.sleep() to be more responsive
                if self.stop_event.wait(timeout=0.5):
                    break

        except KeyboardInterrupt:
            # KeyboardInterrupt during progress display - this is expected
            # the signal handler should have set the stop event
            pass

        console.print()  # new line after progress

    def _process_audio_to_transcript(
        self,
        audio_file: Path,
        base_path: Path,
        stt_engine: str | None = None,
    ) -> tuple[Path, Path] | None:
        """
        process audio file to transcript.

        args:
            audio_file: path to audio file
            base_path: base path for output files
            stt_engine: optional stt engine override

        returns:
            tuple of (text_path, json_path) or None if failed
        """
        console.print("\n[bold]📝 transcription[/bold]")

        # check memory pressure before loading STT model
        if self.config.get("memory", "auto_fallback", True):
            threshold = self.config.get("memory", "warning_threshold", 80)
            # ensure threshold is a float
            check_memory_pressure(float(threshold))

        # use configured engine if not specified
        if not stt_engine:
            stt_engine = self.config.get("models", "stt_engine", "parakeet")
            # map config names to CLI names
            if stt_engine == "faster-whisper":
                stt_engine = "fwhisper"
            elif stt_engine == "mlx-whisper":
                stt_engine = "mlx"
            # parakeet and vosk use their config name directly

        if stt_engine == "parakeet":
            parakeet_model = self.config.get(
                "models", "parakeet_model_name", "mlx-community/parakeet-tdt-0.6b-v3"
            )
            stt_service = ParakeetService(
                model_name=parakeet_model,
            )
        elif stt_engine == "fwhisper":
            stt_model_name = self.config.get("models", "stt_model_name", "large-v3")
            stt_model_path = self.config.expand_path(self.config.get("models", "stt_model_path"))
            stt_service = FasterWhisperService(
                model_path=str(stt_model_path),
                model_name=stt_model_name,
                auto_download=True,
            )
        elif stt_engine == "mlx":
            mlx_model_name = self.config.get(
                "models", "mlx_stt_model_name", "mlx-community/whisper-large-v3-turbo"
            )
            mlx_model_path = self.config.expand_path(
                self.config.get("models", "mlx_stt_model_path")
            )
            stt_service = MlxWhisperService(
                model_name=mlx_model_name,
                model_path=str(mlx_model_path) if mlx_model_path.exists() else None,
                auto_download=True,
            )
        elif stt_engine == "vosk":
            vosk_model_path = self.config.expand_path(self.config.get("models", "vosk_model_path"))
            vosk_spk_model_path = self.config.expand_path(
                self.config.get("models", "vosk_spk_model_path")
            )
            enable_diarization = self.config.get("models", "enable_speaker_diarization", False)
            stt_service = VoskTranscriptionService(
                model_path=str(vosk_model_path),
                spk_model_path=str(vosk_spk_model_path) if vosk_spk_model_path.exists() else None,
                enable_diarization=enable_diarization,
            )
        else:
            # whisper.cpp
            stt_model_path = self.config.get("models", "stt_model_path")
            whisper_cpp_path = self.config.get("models", "whisper_cpp_path", "whisper")
            stt_service = WhisperCppService(
                whisper_cpp_path=whisper_cpp_path,
                model_path=stt_model_path,
            )

        try:
            # check memory before loading STT model
            stt_model_display = self._get_stt_model_name(stt_engine)
            sufficient, avail, needed, msg = check_memory_for_model("stt", stt_model_display)
            if not sufficient:
                console.print(f"[red]{msg}[/red]")
                return None
            elif msg:
                console.print(f"[yellow]{msg}[/yellow]")

            # explicitly load model if supported
            if hasattr(stt_service, "load_model"):
                stt_service.load_model()

            transcript_result = stt_service.transcribe(audio_file)

            # post-STT diarization (works with any engine)
            enable_diarization = self.config.get("models", "enable_speaker_diarization", True)
            diarization_backend = self.config.get("models", "diarization_backend", "sherpa")
            # skip sherpa diarization if using vosk with built-in diarization
            if enable_diarization and diarization_backend == "sherpa" and stt_engine != "vosk":
                try:
                    from meetcap.services.diarization import (
                        SherpaOnnxDiarizationService,
                        assign_speakers,
                    )

                    console.print("\n[bold]🗣️ speaker diarization[/bold]")
                    models_dir = self.config.expand_path(
                        self.config.get("paths", "models_dir", "~/.meetcap/models")
                    )
                    seg_model = str(
                        models_dir / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
                    )
                    emb_model = str(
                        models_dir / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
                    )
                    num_speakers = self.config.get("models", "sherpa_num_speakers", -1)
                    threshold = self.config.get("models", "sherpa_cluster_threshold", 0.90)

                    diar_service = SherpaOnnxDiarizationService(
                        segmentation_model=seg_model,
                        embedding_model=emb_model,
                        num_speakers=int(num_speakers),
                        threshold=float(threshold),
                    )
                    diar_segments = diar_service.diarize(audio_file)
                    transcript_result.segments, transcript_result.speakers = assign_speakers(
                        transcript_result.segments, diar_segments
                    )
                    transcript_result.diarization_enabled = True
                    self._cleanup_service(diar_service, "diarization")
                except ImportError:
                    console.print(
                        "[yellow]⚠ sherpa-onnx not installed, skipping diarization[/yellow]"
                    )
                except FileNotFoundError as e:
                    console.print(f"[yellow]⚠ diarization models not found: {e}[/yellow]")
                    console.print("[yellow]run 'meetcap setup' to download models[/yellow]")
                except Exception as e:
                    console.print(f"[yellow]⚠ diarization failed: {e}[/yellow]")

            text_path, json_path = save_transcript(transcript_result, base_path)

            # explicitly unload model after transcription
            self._cleanup_service(stt_service, "stt")

            return text_path, json_path
        except Exception as e:
            logger.error(f"transcription failed: {e}", exc_info=True)
            console.print(f"[red]transcription failed: {e}[/red]")
            # ensure cleanup on error
            self._cleanup_service(stt_service, "stt")
            return None

    def _process_transcript_to_summary(
        self,
        transcript_path: Path,
        base_path: Path,
        llm_model: str | None = None,
        seed: int | None = None,
    ) -> Path | None:
        """
        process transcript to summary.

        args:
            transcript_path: path to transcript text file
            base_path: base path for output files
            llm_model: optional llm model name override
            seed: optional random seed

        returns:
            path to summary file or None if failed
        """
        console.print("\n[bold]🤖 summarization[/bold]")

        # use provided model name or default from config
        if not llm_model:
            llm_model = self.config.get(
                "models", "llm_model_name", "mlx-community/Qwen3.5-2B-OptiQ-4bit"
            )

        llm_config = self.config.get_section("llm")
        llm_backend = llm_config.get("backend", "mlx-lm")

        if llm_backend == "omlx":
            # use oMLX server for summarization (no in-process model loading)
            # skip memory pressure checks — oMLX manages its own memory
            omlx_url = llm_config.get("omlx_base_url", "http://localhost:8000/v1")
            omlx_key = llm_config.get("omlx_api_key", "")
            omlx_timeout = llm_config.get("omlx_timeout", 300)

            llm_service = OmlxSummarizationService(
                model_name=llm_model,
                base_url=omlx_url,
                temperature=llm_config.get("temperature", 0.4),
                max_tokens=llm_config.get("max_tokens", 4096),
                enable_thinking=llm_config.get("enable_thinking", False),
                thinking_budget=llm_config.get("thinking_budget", 512),
                api_key=omlx_key,
                timeout=int(omlx_timeout),
            )
        else:
            # use in-process mlx-lm / mlx-vlm — check memory pressure first
            if self.config.get("memory", "auto_fallback", True):
                threshold = self.config.get("memory", "warning_threshold", 80)
                check_memory_pressure(float(threshold))

            llm_service = SummarizationService(
                model_name=llm_model,
                temperature=llm_config.get("temperature", 0.4),
                max_tokens=llm_config.get("max_tokens", 4096),
                enable_thinking=llm_config.get("enable_thinking", False),
                thinking_budget=llm_config.get("thinking_budget", 512),
            )

        try:
            # check memory before loading LLM model (skip for oMLX — it manages its own memory)
            if llm_backend != "omlx":
                sufficient, avail, needed, msg = check_memory_for_model("llm", llm_model)
                if not sufficient:
                    console.print(f"[red]{msg}[/red]")
                    return None
                elif msg:
                    console.print(f"[yellow]{msg}[/yellow]")

            # explicitly load model if supported
            if hasattr(llm_service, "load_model"):
                llm_service.load_model()

            # read transcript text
            with open(transcript_path, encoding="utf-8") as f:
                transcript_text = f.read()

            # determine manual notes path
            manual_notes_path = base_path.with_name("notes.md")

            # check if speaker information is available
            has_speaker_info = False
            json_path = base_path.with_suffix(".transcript.json")
            if json_path.exists():
                try:
                    import json

                    with open(json_path, encoding="utf-8") as f:
                        transcript_data = json.load(f)
                        has_speaker_info = transcript_data.get("diarization_enabled", False)
                except Exception:
                    pass  # ignore errors reading JSON

            summary = llm_service.summarize(
                transcript_text,
                has_speaker_info=has_speaker_info,
                manual_notes_path=manual_notes_path,
            )
            summary_path = save_summary(summary, base_path, transcript_text=transcript_text)

            # explicitly unload model after summarization
            self._cleanup_service(llm_service, "llm")

            return summary_path
        except Exception as e:
            logger.error(f"summarization failed: {e}", exc_info=True)
            console.print(f"[red]summarization failed: {e}[/red]")
            # ensure cleanup on error
            self._cleanup_service(llm_service, "llm")
            return None

    def _get_stt_model_name(self, stt_engine: str) -> str:
        """return the configured model name for a given STT engine."""
        engine = stt_engine
        if engine == "faster-whisper":
            engine = "fwhisper"
        elif engine == "mlx-whisper":
            engine = "mlx"

        if engine == "parakeet":
            return self.config.get(
                "models", "parakeet_model_name", "mlx-community/parakeet-tdt-0.6b-v3"
            )
        elif engine == "fwhisper":
            return self.config.get("models", "stt_model_name", "large-v3")
        elif engine == "mlx":
            return self.config.get(
                "models", "mlx_stt_model_name", "mlx-community/whisper-large-v3-turbo"
            )
        elif engine == "vosk":
            return self.config.get("models", "vosk_model_name", "vosk-model-en-us-0.22")
        return "whisper.cpp"

    def _find_recording_file(self, recording_dir: Path) -> Path | None:
        """
        find the recording file in a directory, trying multiple formats.

        args:
            recording_dir: directory to search

        returns:
            path to recording file or None if not found
        """
        # Try each supported format
        for extension in [".opus", ".flac", ".wav"]:
            audio_file = recording_dir / f"recording{extension}"
            if audio_file.exists():
                return audio_file
        return None

    def _process_recording(
        self,
        audio_path: Path,
        stt_engine: str,
        llm_model: str | None,
        seed: int | None,
    ) -> None:
        """
        process recorded audio: transcribe and summarize.

        args:
            audio_path: path to recording directory or audio file
            stt_engine: stt engine to use
            llm_model: optional llm model name
            seed: optional random seed
        """
        # initialize memory monitoring if enabled
        if self.enable_memory_monitoring:
            self.memory_monitor = MemoryMonitor()
            self.memory_monitor.checkpoint("start")

        # pre-flight memory check
        actual_stt = stt_engine or self.config.get("models", "stt_engine", "parakeet")
        stt_model_name = self._get_stt_model_name(actual_stt)
        actual_llm = llm_model or self.config.get(
            "models", "llm_model_name", "mlx-community/Qwen3.5-2B-OptiQ-4bit"
        )
        enable_diar = self.config.get("models", "enable_speaker_diarization", True)
        can_proceed, warning = preflight_memory_check(stt_model_name, actual_llm, enable_diar)
        if warning:
            console.print(f"\n[yellow]warning: {warning}[/yellow]")
        if not can_proceed:
            console.print("[red]aborting: not enough memory to run the pipeline[/red]")
            console.print("[yellow]close other applications and try again[/yellow]")
            return

        # handle both directory and file inputs
        if audio_path.is_dir():
            # called from recording workflow - directory-based
            recording_dir = audio_path
            audio_file = self._find_recording_file(recording_dir)
            if audio_file is None:
                console.print(
                    f"[red]error: no recording file found in {recording_dir}[/red]\n"
                    "[yellow]expected recording.wav, recording.opus, or recording.flac[/yellow]"
                )
                return
            is_recording_workflow = True
        else:
            # called from summarize command - file-based
            recording_dir = None
            audio_file = audio_path
            is_recording_workflow = False

        if not audio_file.exists():
            console.print(f"[red]error: audio file not found: {audio_file}[/red]")
            return

        # base_path for saving files
        if is_recording_workflow:
            base_path = recording_dir / "recording"
        else:
            # for standalone files, use file stem in same directory
            base_path = audio_file.parent / audio_file.stem

        # transcription
        if self.memory_monitor:
            self.memory_monitor.checkpoint("before_stt")

        result = self._process_audio_to_transcript(audio_file, base_path, stt_engine)
        if not result:
            return
        text_path, json_path = result

        if self.memory_monitor:
            self.memory_monitor.checkpoint("after_stt")
            self.memory_monitor.checkpoint("before_llm")

        # summarization
        summary_path = self._process_transcript_to_summary(text_path, base_path, llm_model, seed)
        if not summary_path:
            return

        if self.memory_monitor:
            self.memory_monitor.checkpoint("after_llm")

        # read transcript text for title extraction
        with open(text_path, encoding="utf-8") as f:
            transcript_text = f.read()

        # read summary for title extraction
        with open(summary_path, encoding="utf-8") as f:
            summary = f.read()

        # only organize into directories for recording workflow
        if is_recording_workflow:
            # extract meeting title and rename directory
            console.print("\n[bold]📁 organizing files[/bold]")
            meeting_title = extract_meeting_title(summary, transcript_text)

            # generate final directory name with date and title
            from datetime import datetime

            date_str = datetime.now().strftime("%Y_%b_%d")
            final_dir_name = f"{date_str}_{meeting_title}"
            final_dir_path = recording_dir.parent / final_dir_name

            # rename the temporary directory to final name
            try:
                recording_dir.rename(final_dir_path)
                console.print(f"[green]✓[/green] meeting folder: {final_dir_path.absolute()}")

                # update paths to reflect new location (preserve original file extension)
                audio_file = final_dir_path / audio_file.name
                text_path = final_dir_path / "recording.transcript.txt"
                json_path = final_dir_path / "recording.transcript.json"
                summary_path = final_dir_path / "recording.summary.md"
            except Exception as e:
                console.print(f"[yellow]⚠[/yellow] could not rename folder: {e}")
                console.print(f"[yellow]keeping temporary name: {recording_dir.name}[/yellow]")
                final_dir_path = recording_dir
        else:
            # for standalone files, paths are already set
            final_dir_path = audio_file.parent

        # show final results with absolute paths for easy navigation
        if is_recording_workflow:
            # determine notes file path
            notes_path = final_dir_path / "notes.md"

            console.print(
                Panel(
                    f"[green]✅ recording complete![/green]\n\n"
                    f"[bold]artifacts:[/bold]\n"
                    f"  folder: {final_dir_path.absolute()}\n"
                    f"  audio: {audio_file.absolute()}\n"
                    f"  transcript: {text_path.absolute()}\n"
                    f"  json: {json_path.absolute()}\n"
                    f"  summary: {summary_path.absolute()}\n"
                    f"  notes: {notes_path.absolute()}",
                    title="📦 output files",
                    expand=False,
                )
            )
        else:
            # for standalone files, show simpler output with absolute paths
            # determine notes file path
            notes_path = base_path.with_name("notes.md")

            console.print(
                Panel(
                    f"[green]✅ processing complete![/green]\n\n"
                    f"[bold]output files:[/bold]\n"
                    f"  transcript: {text_path.absolute()}\n"
                    f"  json: {json_path.absolute()}\n"
                    f"  summary: {summary_path.absolute()}\n"
                    f"  notes: {notes_path.absolute()}",
                    title="📦 results",
                    expand=False,
                )
            )

        # show memory report if monitoring enabled
        if self.memory_monitor and self.config.get("memory", "memory_report", False):
            self.memory_monitor.report(detailed=True)

    def _resolve_recording_path(self, path_str: str) -> Path | None:
        """
        resolve recording directory path from user input.

        args:
            path_str: user-provided path (absolute or relative)

        returns:
            resolved path or None if not found
        """
        path = Path(path_str)

        # check if absolute path exists
        if path.is_absolute() and path.exists():
            return path

        # check current directory
        if path.exists():
            return path.resolve()

        # check against configured output directory
        output_dir = self.config.expand_path(self.config.get("paths", "out_dir"))
        candidate = output_dir / path_str
        if candidate.exists():
            return candidate

        # try partial matching for directory names
        if output_dir.exists():
            for item in output_dir.iterdir():
                if item.is_dir() and path_str.lower() in item.name.lower():
                    return item

        return None

    def _reprocess_recording(
        self,
        recording_dir: Path,
        mode: str = "stt",
        stt_engine: str | None = None,
        llm_model: str | None = None,
        skip_confirm: bool = False,
    ) -> None:
        """
        reprocess a recording with new models.

        args:
            recording_dir: path to recording directory
            mode: reprocessing mode ("stt" or "summary")
            stt_engine: optional stt engine override
            llm_model: optional llm model path override
            skip_confirm: skip confirmation prompt
        """
        # validate recording directory - find audio file with any supported format
        audio_file = self._find_recording_file(recording_dir)
        if audio_file is None:
            console.print(
                "[red]error: not a valid recording directory[/red]\n"
                "[yellow]expected recording.wav, recording.opus, or recording.flac[/yellow]"
            )
            return

        # check existing files
        transcript_txt = recording_dir / "recording.transcript.txt"
        transcript_json = recording_dir / "recording.transcript.json"
        summary_md = recording_dir / "recording.summary.md"
        notes_md = recording_dir / "notes.md"

        # backup notes.md for reprocessing
        if notes_md.exists():
            backup_manager_instance = BackupManager()
            backup_manager_instance.create_backup(notes_md)

        # for summary mode, transcript must exist
        if mode == "summary" and not transcript_txt.exists():
            console.print("[red]error: no transcript found to reprocess[/red]")
            console.print("[yellow]run with --mode stt to generate transcript first[/yellow]")
            return

        # resolve actual STT engine being used
        actual_stt_engine = stt_engine
        if not actual_stt_engine and mode == "stt":
            config_stt = self.config.get("models", "stt_engine", "parakeet")
            if config_stt == "faster-whisper":
                actual_stt_engine = "fwhisper"
            elif config_stt == "mlx-whisper":
                actual_stt_engine = "mlx"
            elif config_stt == "vosk":
                actual_stt_engine = "vosk"
            else:
                # parakeet and others use their config name directly
                actual_stt_engine = config_stt

        # get specific model name for display
        stt_display = ""
        if mode == "stt":
            if actual_stt_engine == "parakeet":
                model_name = self.config.get(
                    "models",
                    "parakeet_model_name",
                    "mlx-community/parakeet-tdt-0.6b-v3",
                )
                short_name = model_name.split("/")[-1] if "/" in model_name else model_name
                stt_display = f"parakeet ({short_name})"
            elif actual_stt_engine == "fwhisper":
                model_name = self.config.get("models", "stt_model_name", "large-v3")
                stt_display = f"faster-whisper ({model_name})"
            elif actual_stt_engine == "mlx":
                model_name = self.config.get(
                    "models", "mlx_stt_model_name", "mlx-community/whisper-large-v3-turbo"
                )
                # shorten the display name for mlx models
                short_name = model_name.split("/")[-1] if "/" in model_name else model_name
                stt_display = f"mlx-whisper ({short_name})"
            elif actual_stt_engine == "vosk":
                model_name = self.config.get("models", "vosk_model_name", "vosk-model-en-us-0.22")
                short_name = model_name.replace("vosk-model-", "")
                diarization = (
                    " + speakers"
                    if self.config.get("models", "enable_speaker_diarization", False)
                    else ""
                )
                stt_display = f"vosk ({short_name}{diarization})"
            else:
                stt_display = "whisper.cpp"

            if not stt_engine:
                stt_display += " (default)"

        # resolve actual LLM model being used
        actual_llm_model = llm_model
        if not actual_llm_model:
            actual_llm_model = self.config.get(
                "models", "llm_model_name", "mlx-community/Qwen3.5-2B-OptiQ-4bit"
            )

        # extract model name for display
        llm_display = (
            actual_llm_model.split("/")[-1] if "/" in actual_llm_model else actual_llm_model
        )
        if not llm_model:
            llm_display += " (default)"

        # show confirmation prompt
        if not skip_confirm:
            console.print(
                Panel(
                    f"[bold]📁 recording to reprocess:[/bold] {recording_dir.name}\n"
                    f"   location: {recording_dir.absolute()}\n\n"
                    f"[bold]📋 current files:[/bold]\n"
                    f"   • {audio_file.name} ({audio_file.stat().st_size / 1024 / 1024:.1f} MB)\n"
                    + (
                        f"   • recording.transcript.txt ({transcript_txt.stat().st_size / 1024:.1f} KB)\n"
                        if transcript_txt.exists()
                        else ""
                    )
                    + (
                        f"   • recording.summary.md ({summary_md.stat().st_size / 1024:.1f} KB)\n"
                        if summary_md.exists()
                        else ""
                    )
                    + f"\n[bold]🔄 reprocessing mode:[/bold] {mode.upper()}"
                    + (
                        " (audio → transcript → summary)"
                        if mode == "stt"
                        else " (transcript → summary)"
                    )
                    + (f"\n   stt engine: {stt_display}" if mode == "stt" else "")
                    + f"\n   llm model: {llm_display}\n\n"
                    f"[yellow]⚠️  this will overwrite existing files.[/yellow]\n"
                    f"    backups will be created before processing.",
                    title="reprocess confirmation",
                    expand=False,
                )
            )

            confirm = typer.confirm("continue?", default=False)
            if not confirm:
                console.print("[yellow]reprocessing cancelled[/yellow]")
                return

        # create backup manager
        backup_manager = BackupManager()

        try:
            # create backups
            console.print("\n[bold][1/4] creating backups...[/bold]", end=" ")
            if mode == "stt":
                # backup transcript and summary
                if transcript_txt.exists():
                    backup_manager.create_backup(transcript_txt)
                if transcript_json.exists():
                    backup_manager.create_backup(transcript_json)
                if summary_md.exists():
                    backup_manager.create_backup(summary_md)
            else:
                # backup only summary
                if summary_md.exists():
                    backup_manager.create_backup(summary_md)
            console.print("[green]✓[/green]")

            base_path = recording_dir / "recording"

            if mode == "stt":
                # reprocess from audio
                console.print("[bold][2/4] transcribing audio...[/bold]")
                result = self._process_audio_to_transcript(audio_file, base_path, stt_engine)
                if not result:
                    raise Exception("transcription failed")
                text_path, json_path = result

                console.print("[bold][3/4] generating summary...[/bold]")
                summary_path = self._process_transcript_to_summary(text_path, base_path, llm_model)
                if not summary_path:
                    raise Exception("summarization failed")
            else:
                # reprocess from existing transcript
                console.print(
                    "[bold][2/4] skipping transcription (using existing)[/bold] [green]✓[/green]"
                )
                text_path = transcript_txt

                console.print("[bold][3/4] generating summary...[/bold]")
                summary_path = self._process_transcript_to_summary(text_path, base_path, llm_model)
                if not summary_path:
                    raise Exception("summarization failed")

            # cleanup backups on success
            console.print("[bold][4/4] cleaning up...[/bold]", end=" ")
            backup_manager.cleanup_backups(recording_dir)
            console.print("[green]✓[/green]")

            # show results
            console.print(
                Panel(
                    "[green]✅ reprocessing complete![/green]\n\n"
                    "[bold]updated files:[/bold]\n"
                    + (f"   • transcript: {text_path.absolute()}\n" if mode == "stt" else "")
                    + f"   • summary: {summary_path.absolute()}",
                    title="📦 results",
                    expand=False,
                )
            )

        except Exception as e:
            # restore notes.md from backup on failure
            if notes_md.exists():
                backup_manager.create_backup(notes_md)

            # restore from backups on failure
            console.print(f"\n[red]error during reprocessing: {e}[/red]")
            console.print("[yellow]restoring from backups...[/yellow]")
            backup_manager.restore_all()
            console.print("[yellow]files restored to original state[/yellow]")
            raise


@app.command()
def record(
    device: str | None = typer.Option(
        None,
        "--device",
        "-d",
        help="audio device name or index",
    ),
    out: str | None = typer.Option(
        None,
        "--out",
        "-o",
        help="output directory",
    ),
    rate: int | None = typer.Option(
        None,
        "--rate",
        "-r",
        help="sample rate (hz)",
    ),
    channels: int | None = typer.Option(
        None,
        "--channels",
        "-c",
        help="number of channels",
    ),
    stt: str | None = typer.Option(
        None,
        "--stt",
        help="stt engine: parakeet, fwhisper, mlx, vosk, or whispercpp",
    ),
    llm: str | None = typer.Option(
        None,
        "--llm",
        help="huggingface repo id for llm model",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="random seed for llm",
    ),
    log_file: str | None = typer.Option(
        None,
        "--log-file",
        help="path to log file",
    ),
    auto_stop: int | None = typer.Option(
        None,
        "--auto-stop",
        help="auto stop recording after minutes (30, 60, 90, 120)",
    ),
    audio_format: str | None = typer.Option(
        None,
        "--format",
        "-f",
        help="audio format: wav, opus, or flac (default from config)",
    ),
    opus_bitrate: int | None = typer.Option(
        None,
        "--opus-bitrate",
        help="opus bitrate in kbps (6-510), default optimized for voice",
    ),
    flac_compression: int | None = typer.Option(
        None,
        "--flac-compression",
        help="flac compression level (0-8), higher = smaller file",
    ),
    no_tui: bool = typer.Option(
        False,
        "--no-tui",
        help="disable TUI, use classic output",
    ),
) -> None:
    """start recording a meeting with optional scheduled stop"""

    # setup logging
    if log_file:
        logger.add_file_handler(Path(log_file))

    # load config
    config = Config()

    # Check environment variable if auto_stop is not specified
    if auto_stop is None:
        env_auto_stop = os.environ.get("MEETCAP_RECORDING_AUTO_STOP")
        if env_auto_stop:
            try:
                auto_stop = int(env_auto_stop)
            except ValueError:
                auto_stop = None

    # If auto_stop is not specified, use default from config
    if auto_stop is None:
        auto_stop = config.get("recording", "default_auto_stop", 0)

    # If auto_stop is not specified, prompt user
    if auto_stop is None or auto_stop == 0:
        console.print("[bold]⏱️ Scheduled Stop Options[/bold]\n")
        console.print("1. No automatic stop (manual only)")
        console.print("2. Stop after 30 minutes")
        console.print("3. Stop after 1 hour")
        console.print("4. Stop after 1.5 hours")
        console.print("5. Stop after 2 hours\n")

        choice = typer.prompt("Select option (1-5)", default="1")
        try:
            choice_idx = int(choice)
            if choice_idx == 2:
                auto_stop = 30
            elif choice_idx == 3:
                auto_stop = 60
            elif choice_idx == 4:
                auto_stop = 90
            elif choice_idx == 5:
                auto_stop = 120
            # choice_idx == 1 means no automatic stop
        except ValueError:
            pass  # Default to no automatic stop

    # Validate auto_stop value
    if auto_stop is not None and not validate_auto_stop_time(auto_stop):
        console.print(f"[red]error: invalid auto-stop time {auto_stop} minutes[/red]")
        console.print("[yellow]supported values: 0, 30, 60, 90, 120[/yellow]")
        raise typer.Exit(1)

    # Load audio format settings from config if not provided via CLI
    if audio_format is None:
        audio_format = config.get("audio", "format", "opus")
    if opus_bitrate is None:
        opus_bitrate = config.get("audio", "opus_bitrate", 32)
    if flac_compression is None:
        flac_compression = config.get("audio", "flac_compression_level", 5)

    # Validate audio format
    audio_format_lower = audio_format.lower()
    if audio_format_lower not in ["wav", "opus", "flac"]:
        console.print(f"[red]error: invalid audio format '{audio_format}'[/red]")
        console.print("[yellow]supported formats: wav, opus, flac[/yellow]")
        raise typer.Exit(1)

    # Validate opus bitrate
    if audio_format_lower == "opus":
        if not 6 <= opus_bitrate <= 510:
            console.print(
                f"[red]error: opus bitrate must be between 6-510 kbps (got {opus_bitrate})[/red]"
            )
            raise typer.Exit(1)

    # Validate flac compression
    if audio_format_lower == "flac":
        if not 0 <= flac_compression <= 8:
            console.print(
                f"[red]error: flac compression level must be between 0-8 (got {flac_compression})[/red]"
            )
            raise typer.Exit(1)
    # setup logging
    if log_file:
        logger.add_file_handler(Path(log_file))

    # load config
    config = Config()

    # launch TUI unless --no-tui
    if not no_tui and not os.environ.get("MEETCAP_NO_TUI") and sys.stdout.isatty():
        _launch_tui(
            initial_screen="record",
            record_args={
                "device": device,
                "out": out,
                "rate": rate,
                "channels": channels,
                "stt": stt,
                "llm": llm,
                "auto_stop": auto_stop,
                "audio_format": audio_format_lower,
                "opus_bitrate": opus_bitrate,
            },
        )
        return

    # run orchestrator
    orchestrator = RecordingOrchestrator(config)
    try:
        orchestrator.run(
            device=device,
            output_dir=out,
            sample_rate=rate,
            channels=channels,
            stt_engine=stt,
            llm_model=llm,
            seed=seed,
            auto_stop=auto_stop,
            audio_format=audio_format_lower,
            opus_bitrate=opus_bitrate,
            flac_compression=flac_compression,
        )
    except KeyboardInterrupt:
        # suppress Typer's "Aborted!" message for KeyboardInterrupt
        # the orchestrator's signal handler already managed the graceful stop
        sys.exit(0)


@app.command()
def summarize(
    audio_file: str = typer.Argument(
        ...,
        help="path to audio file (m4a, wav, mp3, etc.)",
    ),
    stt: str | None = typer.Option(
        None,
        "--stt",
        help="stt engine: parakeet, fwhisper, mlx, vosk, or whispercpp",
    ),
    llm: str | None = typer.Option(
        None,
        "--llm",
        help="huggingface repo id for llm model",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="random seed for llm",
    ),
    out: str | None = typer.Option(
        None,
        "--out",
        "-o",
        help="output directory for results",
    ),
    log_file: str | None = typer.Option(
        None,
        "--log-file",
        help="path to log file",
    ),
    no_tui: bool = typer.Option(
        False,
        "--no-tui",
        help="disable TUI, use classic output",
    ),
) -> None:
    """process an existing audio file (transcribe and summarize)"""
    # setup logging
    if log_file:
        logger.add_file_handler(Path(log_file))

    # validate input file
    audio_path = Path(audio_file)
    if not audio_path.exists():
        console.print(f"[red]error: file not found: {audio_file}[/red]")
        sys.exit(1)

    # check if file format is supported
    supported_formats = [".m4a", ".wav", ".mp3", ".mp4", ".aac", ".flac", ".ogg", ".opus", ".webm"]
    if audio_path.suffix.lower() not in supported_formats:
        console.print(f"[red]error: unsupported file format: {audio_path.suffix}[/red]")
        console.print(f"[yellow]supported formats: {', '.join(supported_formats)}[/yellow]")
        sys.exit(1)

    # load config
    config = Config()

    # determine output directory
    if out:
        output_dir = Path(out)
    else:
        # default to same directory as input file
        output_dir = audio_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    # launch TUI unless --no-tui
    if not no_tui and not os.environ.get("MEETCAP_NO_TUI") and sys.stdout.isatty():
        _launch_tui(initial_screen="process", process_file=audio_path)
        return

    # show processing banner
    console.print(
        Panel(
            f"[bold cyan]processing audio file[/bold cyan]\n"
            f"[white]file: {audio_path.name}[/white]\n"
            f"[white]size: {audio_path.stat().st_size / (1024 * 1024):.1f} MB[/white]",
            title="📁 file processing",
            expand=False,
        )
    )

    # process the file (transcribe and summarize)
    orchestrator = RecordingOrchestrator(config)
    orchestrator.processing_complete = False

    try:
        orchestrator._process_recording(
            audio_path=audio_path,
            stt_engine=stt,
            llm_model=llm,
            seed=seed,
        )
        orchestrator.processing_complete = True
        # completion message already shown by _process_recording
    except KeyboardInterrupt:
        console.print("\n[yellow]processing cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]error processing file: {e}[/red]")
        ErrorHandler.handle_runtime_error(e)


@app.command()
def reprocess(
    path: str = typer.Argument(
        ...,
        help="path to recording directory (absolute or relative)",
    ),
    mode: str = typer.Option(
        "stt",
        "--mode",
        "-m",
        help="reprocessing mode: 'stt' (audio→transcript→summary) or 'summary' (transcript→summary)",
    ),
    stt: str | None = typer.Option(
        None,
        "--stt",
        help="stt engine override: 'parakeet', 'fwhisper', 'mlx', 'vosk', or 'whisper.cpp'",
    ),
    llm: str | None = typer.Option(
        None,
        "--llm",
        help="huggingface repo id for llm model",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="skip confirmation prompt",
    ),
    no_tui: bool = typer.Option(
        False,
        "--no-tui",
        help="disable TUI, use classic output",
    ),
) -> None:
    """reprocess a recording with different models"""
    config = Config()
    orchestrator = RecordingOrchestrator(config)

    # validate mode
    if mode not in ["stt", "summary"]:
        console.print(f"[red]error: invalid mode '{mode}'[/red]")
        console.print("[yellow]use --mode stt or --mode summary[/yellow]")
        raise typer.Exit(1)

    # validate stt engine if provided
    if stt and stt not in ["parakeet", "fwhisper", "mlx", "vosk", "whisper.cpp"]:
        console.print(f"[red]error: invalid stt engine '{stt}'[/red]")
        console.print("[yellow]use --stt parakeet, fwhisper, mlx, vosk, or whisper.cpp[/yellow]")
        raise typer.Exit(1)

    # resolve recording path
    recording_dir = orchestrator._resolve_recording_path(path)
    if not recording_dir:
        console.print(f"[red]error: recording directory not found: {path}[/red]")
        console.print("\n[yellow]hints:[/yellow]")
        console.print("  • use absolute path: /path/to/recording")
        console.print("  • use relative path: 2025_Jan_15_TeamMeeting")
        console.print("  • check configured output directory:")
        console.print(f"    {config.expand_path(config.get('paths', 'out_dir'))}")
        raise typer.Exit(1)

    # launch TUI unless --no-tui
    if not no_tui and not os.environ.get("MEETCAP_NO_TUI") and sys.stdout.isatty():
        # find audio file in the recording directory
        audio_file_path = None
        for ext in [".opus", ".wav", ".flac"]:
            audio_files = list(recording_dir.glob(f"*{ext}"))
            if audio_files:
                audio_file_path = audio_files[0]
                break
        if audio_file_path:
            _launch_tui(initial_screen="process", process_file=audio_file_path)
            return

    try:
        orchestrator._reprocess_recording(
            recording_dir=recording_dir,
            mode=mode,
            stt_engine=stt,
            llm_model=llm,
            skip_confirm=yes,
        )
    except Exception as e:
        console.print(f"[red]reprocessing failed: {e}[/red]")
        ErrorHandler.handle_runtime_error(e)


@app.command()
def devices() -> None:
    """list available audio input devices"""
    console.print("[bold]🎤 audio input devices[/bold]\n")

    devices = list_audio_devices()
    if devices:
        print_devices(devices)
    else:
        console.print("[red]no audio devices found[/red]")
        console.print("\n[yellow]troubleshooting:[/yellow]")
        console.print("  • ensure ffmpeg is installed: brew install ffmpeg")
        console.print("  • grant microphone permission to your terminal")
        console.print("  • check audio midi setup for device configuration")


@app.command()
def setup(
    force_download: bool = typer.Option(
        False,
        "--force-download",
        help="force re-download all models even if they exist",
    ),
) -> None:
    """interactive setup wizard for first-time configuration"""
    console.print(
        Panel(
            "[bold cyan]meetcap setup wizard[/bold cyan]\n"
            "[white]this will guide you through permissions and model downloads[/white]",
            title="🛠️ initial setup",
            expand=False,
        )
    )

    config = Config()
    models_dir = config.expand_path(config.get("paths", "models_dir", "~/.meetcap/models"))

    # step 1: check ffmpeg
    console.print("\n[bold]step 1: checking dependencies[/bold]")
    import subprocess

    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=2)
        if result.returncode == 0:
            console.print("[green]✓[/green] ffmpeg is installed")
        else:
            console.print("[red]✗[/red] ffmpeg error")
            console.print("[yellow]please install: brew install ffmpeg[/yellow]")
            return
    except FileNotFoundError:
        console.print("[red]✗[/red] ffmpeg not found")
        console.print("[yellow]please install: brew install ffmpeg[/yellow]")
        return

    # step 2: test microphone permission
    console.print("\n[bold]step 2: microphone permission[/bold]")
    console.print("[cyan]testing microphone access...[/cyan]")
    console.print("[yellow]⚠ macos may prompt for microphone permission[/yellow]")

    devices = list_audio_devices()
    if not devices:
        console.print("[red]✗[/red] no audio devices found")
        console.print("[yellow]grant microphone permission in system preferences[/yellow]")
        console.print("system preferences → privacy & security → microphone")
        return

    # try a brief recording to trigger permission dialog
    console.print("[cyan]attempting test recording to verify permissions...[/cyan]")

    # use a temporary directory for the test recording
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        recorder = AudioRecorder(output_dir=temp_path)
        try:
            recorder.start_recording(
                device_index=devices[0].index,
                device_name=devices[0].name,
            )
            time.sleep(2)  # record for 2 seconds
            recorder.stop_recording()

            # if we got here, permissions are granted
            console.print("[green]✓[/green] microphone permission granted")
            console.print(f"  detected {len(devices)} audio device(s)")

            # temp directory and contents will be automatically cleaned up
        except Exception as e:
            console.print("[red]✗[/red] microphone permission denied or error")
            console.print(f"  error: {e}")
            console.print(
                "[yellow]grant permission in system preferences and run setup again[/yellow]"
            )
            return

    # step 3: test hotkey permission
    console.print("\n[bold]step 3: input monitoring permission (for hotkeys)[/bold]")
    console.print("[cyan]testing hotkey functionality...[/cyan]")
    console.print("[yellow]⚠ grant input monitoring permission if prompted[/yellow]")

    # create a simple test for hotkey
    test_triggered = threading.Event()

    def test_callback():
        test_triggered.set()

    hotkey_mgr = HotkeyManager(test_callback)
    hotkey_mgr.start("<cmd>+<shift>+t")  # test hotkey

    console.print("[cyan]press ⌘⇧T to test hotkey (or wait 5 seconds to skip)...[/cyan]")

    if test_triggered.wait(timeout=5.0):
        console.print("[green]✓[/green] hotkey permission granted")
    else:
        console.print("[yellow]⚠[/yellow] hotkey not detected (permission may be needed)")
        console.print("  grant input monitoring in system preferences if you want hotkey support")
        console.print("  you can still use Ctrl-C to stop recordings")

    hotkey_mgr.stop()

    # step 4: select and download speech-to-text model
    console.print("\n[bold]step 4: select speech-to-text (STT) engine[/bold]")

    # check if running on apple silicon
    import platform

    is_apple_silicon = platform.processor() == "arm"

    if is_apple_silicon:
        console.print(
            "[cyan]detected Apple Silicon - mlx-whisper available for better performance[/cyan]"
        )
        stt_engines = [
            {
                "key": "parakeet",
                "name": "Parakeet TDT (recommended — 16x faster)",
                "default_model": "mlx-community/parakeet-tdt-0.6b-v3",
            },
            {
                "key": "mlx",
                "name": "MLX Whisper (Apple Silicon)",
                "default_model": "mlx-community/whisper-large-v3-turbo",
            },
            {
                "key": "faster",
                "name": "Faster Whisper (universal compatibility)",
                "default_model": "large-v3",
            },
            {
                "key": "vosk",
                "name": "Vosk (offline, with speaker identification)",
                "default_model": "vosk-model-en-us-0.22",
            },
        ]

        console.print("\n[cyan]available stt engines:[/cyan]")
        for i, engine in enumerate(stt_engines, 1):
            console.print(f"  {i}. [bold]{engine['name']}[/bold]")

        engine_choice = typer.prompt("\nselect engine (1-4)", default="1")
        try:
            engine_idx = int(engine_choice) - 1
            if 0 <= engine_idx < len(stt_engines):
                selected_engine = stt_engines[engine_idx]
            else:
                selected_engine = stt_engines[0]
        except ValueError:
            selected_engine = stt_engines[0]
    else:
        stt_engines = [
            {"key": "faster", "name": "Faster Whisper (recommended)", "default_model": "large-v3"},
            {
                "key": "vosk",
                "name": "Vosk (offline, with speaker identification)",
                "default_model": "vosk-model-en-us-0.22",
            },
        ]

        console.print("\n[cyan]available stt engines:[/cyan]")
        for i, engine in enumerate(stt_engines, 1):
            console.print(f"  {i}. [bold]{engine['name']}[/bold]")

        engine_choice = typer.prompt("\nselect engine (1-2)", default="1")
        try:
            engine_idx = int(engine_choice) - 1
            if 0 <= engine_idx < len(stt_engines):
                selected_engine = stt_engines[engine_idx]
            else:
                selected_engine = stt_engines[0]
        except ValueError:
            selected_engine = stt_engines[0]

    console.print(f"\n[cyan]selected engine: {selected_engine['name']}[/cyan]")

    if selected_engine["key"] == "parakeet":
        console.print("[cyan]downloading parakeet model (first use ~2.5 GB)...[/cyan]")
        try:
            from huggingface_hub import hf_hub_download

            model_name = selected_engine["default_model"]
            if force_download:
                console.print("[cyan]force re-downloading parakeet model...[/cyan]")
            hf_hub_download(model_name, "config.json", force_download=force_download)
            hf_hub_download(model_name, "model.safetensors", force_download=force_download)
            console.print("[green]✓[/green] parakeet model ready")
        except Exception as e:
            console.print(f"[red]✗[/red] parakeet model download failed: {e}")
            console.print("[yellow]check your internet connection and try again[/yellow]")
            return

        # update config
        config.config["models"]["stt_engine"] = "parakeet"
        config.config["models"]["parakeet_model_name"] = selected_engine["default_model"]
        config.save()

    elif selected_engine["key"] == "vosk":
        # vosk models
        vosk_models = [
            {
                "name": "vosk-model-small-en-us-0.15",
                "desc": "Fast, lower accuracy",
                "size": "~507MB",
            },
            {"name": "vosk-model-en-us-0.22", "desc": "Balanced (recommended)", "size": "~1.8GB"},
            {"name": "vosk-model-en-us-0.42-gigaspeech", "desc": "Best accuracy", "size": "~3.3GB"},
        ]

        console.print("\n[cyan]available vosk models:[/cyan]")
        for i, model in enumerate(vosk_models, 1):
            console.print(
                f"  {i}. [bold]{model['name'].replace('vosk-model-', '')}[/bold] - {model['desc']} ({model['size']})"
            )

        choice = typer.prompt("\nselect model (1-3)", default="2")
        try:
            model_idx = int(choice) - 1
            if 0 <= model_idx < len(vosk_models):
                vosk_model_name = vosk_models[model_idx]["name"]
            else:
                vosk_model_name = vosk_models[1]["name"]
        except ValueError:
            vosk_model_name = vosk_models[1]["name"]

        console.print(f"\n[cyan]selected: {vosk_model_name.replace('vosk-model-', '')}[/cyan]")

        # download vosk model if needed
        if not force_download and verify_vosk_model(vosk_model_name, models_dir / "vosk"):
            console.print("[green]✓[/green] vosk model already installed")
        else:
            console.print("[cyan]downloading vosk model...[/cyan]")
            model_path = ensure_vosk_model(vosk_model_name, models_dir / "vosk")

            if model_path:
                console.print("[green]✓[/green] vosk model ready")
            else:
                console.print("[red]✗[/red] vosk download failed")
                console.print("[yellow]check your internet connection and try again[/yellow]")
                return

        # ask about speaker diarization
        enable_diarization = typer.confirm(
            "\nenable speaker identification (diarization)?", default=True
        )

        if enable_diarization:
            console.print("[cyan]downloading speaker model...[/cyan]")
            spk_model_path = ensure_vosk_spk_model(models_dir / "vosk")
            if spk_model_path:
                console.print("[green]✓[/green] speaker model ready")
            else:
                console.print(
                    "[yellow]⚠ speaker model download failed, diarization will be disabled[/yellow]"
                )
                enable_diarization = False

        # update config
        config.config["models"]["stt_engine"] = "vosk"
        config.config["models"]["vosk_model_name"] = vosk_model_name
        config.config["models"]["vosk_model_path"] = str(models_dir / "vosk" / vosk_model_name)
        config.config["models"]["vosk_spk_model_path"] = str(
            models_dir / "vosk" / "vosk-model-spk-0.4"
        )
        config.config["models"]["enable_speaker_diarization"] = enable_diarization
        config.save()

    elif selected_engine["key"] == "mlx":
        # mlx-whisper models
        mlx_models = [
            {
                "name": "mlx-community/whisper-large-v3-turbo",
                "desc": "Fast and accurate (recommended)",
                "size": "~1.5GB",
            },
            {
                "name": "mlx-community/whisper-large-v3-mlx",
                "desc": "Most accurate",
                "size": "~1.5GB",
            },
            {
                "name": "mlx-community/whisper-small-mlx",
                "desc": "Smallest, fastest",
                "size": "~466MB",
            },
        ]

        console.print("\n[cyan]available mlx-whisper models:[/cyan]")
        for i, model in enumerate(mlx_models, 1):
            console.print(
                f"  {i}. [bold]{model['name'].split('/')[-1]}[/bold] - {model['desc']} ({model['size']})"
            )

        choice = typer.prompt("\nselect model (1-3)", default="1")
        try:
            model_idx = int(choice) - 1
            if 0 <= model_idx < len(mlx_models):
                mlx_model_name = mlx_models[model_idx]["name"]
            else:
                mlx_model_name = mlx_models[0]["name"]
        except ValueError:
            mlx_model_name = mlx_models[0]["name"]

        console.print(f"\n[cyan]selected: {mlx_model_name.split('/')[-1]}[/cyan]")

        # verify/download if needed
        if not force_download and verify_mlx_whisper_model(mlx_model_name, models_dir):
            console.print(
                f"[green]✓[/green] mlx-whisper {mlx_model_name.split('/')[-1]} already installed"
            )
        else:
            console.print(
                f"[cyan]downloading mlx-whisper {mlx_model_name.split('/')[-1]}...[/cyan]"
            )
            console.print("[dim]this may take several minutes[/dim]")

            model_path = ensure_mlx_whisper_model(mlx_model_name, models_dir)

            if model_path:
                console.print("[green]✓[/green] mlx-whisper model ready")
            else:
                console.print("[red]✗[/red] mlx-whisper download failed")
                console.print("[yellow]check your internet connection and try again[/yellow]")
                return

        # update config
        config.config["models"]["stt_engine"] = "mlx-whisper"
        config.config["models"]["mlx_stt_model_name"] = mlx_model_name
        config.save()

    else:
        # faster-whisper models
        whisper_models = [
            {"name": "large-v3", "desc": "Most accurate, slower (default)", "size": "~1.5GB"},
            {
                "name": "large-v3-turbo",
                "desc": "Faster than v3, slightly less accurate",
                "size": "~1.5GB",
            },
            {"name": "small", "desc": "Fast, good for quick transcripts", "size": "~466MB"},
        ]

        console.print("\n[cyan]available whisper models:[/cyan]")
        for i, model in enumerate(whisper_models, 1):
            console.print(
                f"  {i}. [bold]{model['name']}[/bold] - {model['desc']} ({model['size']})"
            )

        choice = typer.prompt("\nselect model (1-3)", default="1")
        try:
            model_idx = int(choice) - 1
            if 0 <= model_idx < len(whisper_models):
                stt_model_name = whisper_models[model_idx]["name"]
            else:
                stt_model_name = "large-v3"
        except ValueError:
            stt_model_name = "large-v3"

        console.print(f"\n[cyan]selected: {stt_model_name}[/cyan]")

        # download if needed
        if not force_download and verify_whisper_model(stt_model_name, models_dir):
            console.print(f"[green]✓[/green] whisper {stt_model_name} already installed")
        else:
            console.print(f"[cyan]downloading whisper {stt_model_name}...[/cyan]")
            console.print("[dim]this may take several minutes[/dim]")

            model_path = ensure_whisper_model(stt_model_name, models_dir)

            if model_path:
                console.print("[green]✓[/green] whisper model downloaded")
            else:
                console.print("[red]✗[/red] whisper download failed")
                console.print("[yellow]check your internet connection and try again[/yellow]")
                return

        # update config
        config.config["models"]["stt_engine"] = "faster-whisper"
        config.config["models"]["stt_model_name"] = stt_model_name
        config.config["models"]["stt_model_path"] = f"~/.meetcap/models/whisper-{stt_model_name}"
        config.save()

    # step 5: configure output directory
    console.print("\n[bold]step 5: configure output directory[/bold]")
    console.print("[cyan]where should recordings be saved?[/cyan]")

    current_out_dir = config.get("paths", "out_dir", "~/Recordings/meetcap")
    console.print(f"\n[dim]current: {current_out_dir}[/dim]")

    new_out_dir = typer.prompt(
        "\noutput directory path", default=current_out_dir, show_default=True
    )

    # expand and validate the path
    expanded_path = config.expand_path(new_out_dir)
    try:
        expanded_path.mkdir(parents=True, exist_ok=True)
        config.config["paths"]["out_dir"] = new_out_dir
        config.save()
        console.print(f"[green]✓[/green] output directory set to: {new_out_dir}")
    except Exception as e:
        console.print(f"[yellow]⚠[/yellow] could not create directory: {e}")
        console.print(f"[yellow]keeping current directory: {current_out_dir}[/yellow]")

    # step 6: download diarization models
    console.print("\n[bold]step 6: download speaker diarization models[/bold]")
    console.print("[cyan]sherpa-onnx diarization requires two small models (~43 MB total)[/cyan]")

    seg_model_dir = models_dir / "sherpa-onnx-pyannote-segmentation-3-0"
    seg_model_path = seg_model_dir / "model.onnx"
    emb_model_path = models_dir / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

    if not force_download and seg_model_path.exists() and emb_model_path.exists():
        console.print("[green]✓[/green] diarization models already downloaded")
    else:
        download_diar = typer.confirm("download diarization models?", default=True)
        if download_diar:
            import tarfile

            # download segmentation model
            if force_download or not seg_model_path.exists():
                console.print("[cyan]downloading segmentation model (~5 MB)...[/cyan]")
                seg_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
                seg_archive = models_dir / "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
                try:
                    urllib.request.urlretrieve(seg_url, seg_archive)
                    with tarfile.open(seg_archive, "r:bz2") as tar:
                        tar.extractall(path=models_dir)
                    seg_archive.unlink()
                    console.print("[green]✓[/green] segmentation model ready")
                except Exception as e:
                    console.print(f"[red]✗[/red] segmentation model download failed: {e}")

            # download embedding model
            if force_download or not emb_model_path.exists():
                console.print("[cyan]downloading embedding model (~38 MB)...[/cyan]")
                emb_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
                try:
                    urllib.request.urlretrieve(emb_url, emb_model_path)
                    console.print("[green]✓[/green] embedding model ready")
                except Exception as e:
                    console.print(f"[red]✗[/red] embedding model download failed: {e}")
        else:
            console.print("[yellow]⚠[/yellow] diarization models not downloaded")
            console.print("[yellow]speaker diarization will not be available[/yellow]")

    config.config["models"]["enable_speaker_diarization"] = True
    config.config["models"]["diarization_backend"] = "sherpa"
    config.save()

    # step 7: select and download llm model
    console.print("\n[bold]step 7: select llm (summarization) model[/bold]")

    llm_models = [
        {
            "repo": "mlx-community/Qwen3.5-2B-OptiQ-4bit",
            "name": "Qwen3.5-4B",
            "desc": "Best for meeting summaries (default)",
            "size": "~2.9GB",
        },
        {
            "repo": "mlx-community/Qwen3.5-9B-MLX-4bit",
            "name": "Qwen3.5-9B",
            "desc": "Larger model, more capable",
            "size": "~5.6GB",
        },
    ]

    console.print("\n[cyan]available llm models:[/cyan]")
    for i, model in enumerate(llm_models, 1):
        console.print(f"  {i}. [bold]{model['name']}[/bold] - {model['desc']} ({model['size']})")

    choice = typer.prompt("\nselect model (1-2)", default="1")
    try:
        model_idx = int(choice) - 1
        if 0 <= model_idx < len(llm_models):
            llm_choice = llm_models[model_idx]
        else:
            llm_choice = llm_models[0]
    except ValueError:
        llm_choice = llm_models[0]

    console.print(f"\n[cyan]selected: {llm_choice['name']}[/cyan]")

    repo = llm_choice["repo"]

    # check if already available
    if not force_download and verify_mlx_llm_model(repo):
        console.print(f"[green]✓[/green] {llm_choice['name']} already installed")
    else:
        console.print(f"[cyan]downloading {llm_choice['name']} ({llm_choice['size']})...[/cyan]")
        console.print("[yellow]⚠ this may take several minutes[/yellow]")

        # ask for confirmation
        if typer.confirm("proceed with download?"):
            success = ensure_mlx_llm_model(repo)

            if success:
                console.print(f"[green]✓[/green] {llm_choice['name']} downloaded")
            else:
                console.print(f"[red]✗[/red] {llm_choice['name']} download failed")
                console.print("[yellow]check your internet connection and try again[/yellow]")
                return
        else:
            console.print("[yellow]skipped llm download (summarization will not work)[/yellow]")

    # update config with selected model
    config.config["models"]["llm_model_name"] = repo
    config.save()

    # final summary
    console.print(
        Panel(
            "[green]✅ setup complete![/green]\n\n"
            f"output directory: {new_out_dir}\n"
            "you're ready to start recording meetings:\n"
            "[cyan]meetcap record[/cyan]",
            title="🎉 success",
            expand=False,
        )
    )


@app.command()
def verify() -> None:
    """quick verification of system setup"""
    console.print("[bold]🔍 system verification[/bold]\n")

    config = Config()
    checks = []

    # check ffmpeg
    import subprocess

    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=2,
        )
        if result.returncode == 0:
            checks.append(("ffmpeg", "✅ installed", "green"))
        else:
            checks.append(("ffmpeg", "❌ error", "red"))
    except FileNotFoundError:
        checks.append(("ffmpeg", "❌ not found", "red"))
    except Exception:
        checks.append(("ffmpeg", "⚠️ unknown", "yellow"))

    # check audio devices
    devices = list_audio_devices()
    if devices:
        aggregate_found = any(d.is_aggregate for d in devices)
        if aggregate_found:
            checks.append(
                ("audio devices", f"✅ {len(devices)} found (aggregate detected)", "green")
            )
        else:
            checks.append(("audio devices", f"⚠️ {len(devices)} found (no aggregate)", "yellow"))
    else:
        checks.append(("audio devices", "❌ none found", "red"))

    # check microphone permission
    if PermissionChecker.check_microphone_permission():
        checks.append(("microphone", "✅ permission likely granted", "green"))
    else:
        checks.append(("microphone", "⚠️ permission unknown", "yellow"))

    # check stt models (no download)
    stt_model_name = config.get("models", "stt_model_name", "large-v3")
    mlx_model_name = config.get(
        "models", "mlx_stt_model_name", "mlx-community/whisper-large-v3-turbo"
    )
    models_dir = config.expand_path(config.get("paths", "models_dir", "~/.meetcap/models"))

    # check faster-whisper
    if verify_whisper_model(stt_model_name, models_dir):
        checks.append(("faster-whisper", f"✅ {stt_model_name} ready", "green"))
    else:
        checks.append(("faster-whisper", f"❌ {stt_model_name} not found", "red"))

    # check mlx-whisper (only on Apple Silicon)
    import platform

    if platform.processor() == "arm":
        if verify_mlx_whisper_model(mlx_model_name, models_dir):
            checks.append(("mlx-whisper", f"✅ {mlx_model_name.split('/')[-1]} ready", "green"))
        else:
            checks.append(("mlx-whisper", f"❌ {mlx_model_name.split('/')[-1]} not found", "red"))
    else:
        checks.append(("mlx-whisper", "⚠️ requires Apple Silicon", "yellow"))

    # check mlx llm model (no download)
    llm_model_name = config.get("models", "llm_model_name", "mlx-community/Qwen3.5-2B-OptiQ-4bit")
    if verify_mlx_llm_model(llm_model_name):
        short_name = llm_model_name.split("/")[-1] if "/" in llm_model_name else llm_model_name
        checks.append(("llm model", f"✅ {short_name} ready", "green"))
    else:
        checks.append(("llm model", "❌ mlx llm model not found", "red"))

    # check output directory
    out_dir = config.expand_path(config.get("paths", "out_dir"))
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        checks.append(("output dir", "✅ writable", "green"))
    except Exception as e:
        checks.append(("output dir", f"❌ error: {e}", "red"))

    # display results
    table = Table(show_header=True, header_style="bold")
    table.add_column("component", style="cyan")
    table.add_column("status")

    all_good = True
    for component, status, color in checks:
        table.add_row(component, f"[{color}]{status}[/{color}]")
        if color == "red":
            all_good = False

    console.print(table)

    if not all_good:
        console.print("\n[yellow]⚠ some components are missing or need attention[/yellow]")
        console.print("run 'meetcap setup' to install models and configure permissions")
    else:
        console.print("\n[green]✅ all checks passed![/green]")
        console.print("ready to record with: meetcap record")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="show version",
    ),
) -> None:
    """meetcap - offline meeting recorder & summarizer for macos"""
    if version:
        console.print(f"meetcap v{__version__}")
        raise typer.Exit()
    # if no subcommand, launch TUI
    if ctx.invoked_subcommand is None:
        _launch_tui()


def _launch_tui(
    initial_screen: str = "home",
    record_args: dict | None = None,
    process_file: "Path | None" = None,
) -> None:
    """launch the textual TUI application."""
    if not sys.stdout.isatty() or os.environ.get("MEETCAP_NO_TUI"):
        return
    try:  # pragma: no cover
        from meetcap.tui.app import MeetcapApp

        tui = MeetcapApp(
            initial_screen=initial_screen,
            record_args=record_args,
            process_file=process_file,
        )
        tui.run()
        raise typer.Exit()
    except ImportError:  # pragma: no cover
        console.print("[yellow]TUI not available. Install with: pip install meetcap[tui][/yellow]")


if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        ErrorHandler.handle_general_error(e)
