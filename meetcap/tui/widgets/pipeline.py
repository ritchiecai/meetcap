from __future__ import annotations

import time
from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Label, ProgressBar

# spinner frames cycled while a stage is active so the user can see something
# is moving even when the underlying model has no progress callback.
_SPINNER_FRAMES = ["\u2839", "\u2819", "\u2812", "\u280a", "\u2828", "\u2820"]


@dataclass
class StageInfo:
    """information about a pipeline stage."""

    name: str
    label: str
    status: str = "pending"  # pending, active, done, error
    progress: float = 0.0
    timing: float = 0.0
    detail: str = ""


class StageWidget(Widget):
    """individual pipeline stage display."""

    def __init__(self, stage: StageInfo, **kwargs) -> None:
        super().__init__(**kwargs)
        self._stage = stage
        self._active_started_at: float = 0.0
        self._spinner_idx: int = 0
        self._tick_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        yield Label(
            self._format_label(),
            id=f"stage-label-{self._stage.name}",
        )
        yield ProgressBar(
            total=100,
            show_eta=False,
            id=f"stage-bar-{self._stage.name}",
        )

    def _format_label(self) -> str:
        s = self._stage
        if s.status == "active":
            # use a moving spinner frame instead of a static circle so
            # the user can see at a glance the stage is alive.
            status_icon = _SPINNER_FRAMES[self._spinner_idx % len(_SPINNER_FRAMES)]
        else:
            status_icon = {
                "pending": "\u25cb",
                "done": "\u2713",
                "error": "\u2717",
            }.get(s.status, "\u25cb")

        # while active we surface the elapsed seconds in real-time so the
        # user knows the work is still happening, not stuck.
        if s.status == "active" and self._active_started_at > 0:
            elapsed = time.time() - self._active_started_at
            timing = f"  {elapsed:.0f}s"
        elif s.timing > 0:
            timing = f"  {s.timing:.1f}s"
        else:
            timing = ""

        detail = f"  {s.detail}" if s.detail else ""
        return f"{status_icon}  {s.label}{timing}{detail}"

    def _on_tick(self) -> None:
        """called every second while the stage is active."""
        self._spinner_idx += 1
        try:
            label = self.query_one(f"#stage-label-{self._stage.name}", Label)
            label.update(self._format_label())
        except Exception:
            pass

    def _start_ticking(self) -> None:
        """start the spinner / elapsed timer."""
        self._active_started_at = time.time()
        self._spinner_idx = 0
        if self._tick_timer is None:
            try:
                self._tick_timer = self.set_interval(1.0, self._on_tick)
            except Exception:
                self._tick_timer = None

    def _stop_ticking(self) -> None:
        """stop the spinner timer (if running)."""
        if self._tick_timer is not None:
            try:
                self._tick_timer.stop()
            except Exception:
                pass
            self._tick_timer = None
        self._active_started_at = 0.0

    def update_stage(
        self,
        status: str,
        progress: float = 0.0,
        timing: float = 0.0,
        detail: str = "",
    ) -> None:
        prev_status = self._stage.status
        self._stage.status = status
        self._stage.progress = progress
        self._stage.timing = timing
        self._stage.detail = detail

        # manage spinner / elapsed timer based on transitions.
        if status == "active" and prev_status != "active":
            self._start_ticking()
        elif status != "active" and prev_status == "active":
            self._stop_ticking()

        try:
            label = self.query_one(f"#stage-label-{self._stage.name}", Label)
            label.update(self._format_label())
            bar = self.query_one(f"#stage-bar-{self._stage.name}", ProgressBar)
            if status == "active":
                # indeterminate / pulsing bar — textual renders a
                # continuously moving animation when total is None.
                bar.update(total=None)
            elif status == "done":
                bar.update(total=100, progress=100)
            elif status == "error":
                bar.update(total=100, progress=0)
            else:
                bar.update(total=100, progress=progress)
        except Exception:
            pass


class PipelineProgress(Widget):
    """multi-stage processing pipeline progress display."""

    DEFAULT_STAGES = [
        StageInfo(name="stt", label="STT"),
        StageInfo(name="diarization", label="Diarization"),
        StageInfo(name="summarization", label="Summarization"),
        StageInfo(name="organize", label="File Organization"),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._stages: dict[str, StageInfo] = {}
        self._widgets: dict[str, StageWidget] = {}

    def compose(self) -> ComposeResult:
        with Vertical(id="pipeline-stages"):
            for stage_info in self.DEFAULT_STAGES:
                info = StageInfo(name=stage_info.name, label=stage_info.label)
                self._stages[info.name] = info
                widget = StageWidget(info, classes="stage-widget")
                self._widgets[info.name] = widget
                yield widget

    def update_stage(
        self,
        name: str,
        status: str,
        progress: float = 0.0,
        timing: float = 0.0,
        detail: str = "",
    ) -> None:
        if name in self._widgets:
            self._widgets[name].update_stage(status, progress, timing, detail)

    def reset(self) -> None:
        for _name, widget in self._widgets.items():
            widget.update_stage("pending", 0.0, 0.0, "")
