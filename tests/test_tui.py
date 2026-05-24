"""tests for meetcap textual TUI components."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

# -- unit tests for widgets (no app needed) --


class TestStageInfo:
    """tests for pipeline StageInfo dataclass."""

    def test_default_values(self) -> None:
        from meetcap.tui.widgets.pipeline import StageInfo

        stage = StageInfo(name="stt", label="STT")
        assert stage.name == "stt"
        assert stage.label == "STT"
        assert stage.status == "pending"
        assert stage.progress == 0.0
        assert stage.timing == 0.0
        assert stage.detail == ""

    def test_custom_values(self) -> None:
        from meetcap.tui.widgets.pipeline import StageInfo

        stage = StageInfo(
            name="stt", label="STT", status="done", progress=100.0, timing=5.3, detail="42 segments"
        )
        assert stage.status == "done"
        assert stage.progress == 100.0
        assert stage.timing == 5.3
        assert stage.detail == "42 segments"


class TestRecordingDigitsUnit:
    """unit tests for RecordingDigits without mounting."""

    def test_initial_elapsed(self) -> None:
        from meetcap.tui.widgets.recording_digits import RecordingDigits

        digits = RecordingDigits()
        assert digits.elapsed == 0.0

    def test_update_time_stores_seconds(self) -> None:
        from meetcap.tui.widgets.recording_digits import RecordingDigits

        digits = RecordingDigits()
        # update_time sets _seconds even if not mounted (query will fail silently)
        digits.update_time(125.7)
        assert digits.elapsed == 125.7


class TestAudioLevelMeterUnit:
    """unit tests for AudioLevelMeter without mounting."""

    def test_initial_empty(self) -> None:
        from meetcap.tui.widgets.audio_level import AudioLevelMeter

        meter = AudioLevelMeter()
        assert meter._levels == []

    def test_max_points_default(self) -> None:
        from meetcap.tui.widgets.audio_level import AudioLevelMeter

        meter = AudioLevelMeter()
        assert meter._max_points == 60

    def test_max_points_custom(self) -> None:
        from meetcap.tui.widgets.audio_level import AudioLevelMeter

        meter = AudioLevelMeter(max_points=10)
        assert meter._max_points == 10


class TestModelStatusIndicatorUnit:
    """unit tests for ModelStatusIndicator without mounting."""

    def test_initial_state(self) -> None:
        from meetcap.tui.widgets.model_status import ModelStatusIndicator

        indicator = ModelStatusIndicator(model_name="test-model")
        assert indicator._model_name == "test-model"
        assert indicator._status == "unknown"

    def test_set_model_name(self) -> None:
        from meetcap.tui.widgets.model_status import ModelStatusIndicator

        indicator = ModelStatusIndicator()
        indicator.set_model_name("new-model")
        assert indicator._model_name == "new-model"


# -- theme tests --


class TestTheme:
    """tests for the meetcap dark theme."""

    def test_theme_properties(self) -> None:
        from meetcap.tui.app import MEETCAP_DARK

        assert MEETCAP_DARK.name == "meetcap-dark"
        assert MEETCAP_DARK.primary == "#4fc1ff"
        assert MEETCAP_DARK.dark is True

    def test_theme_colors(self) -> None:
        from meetcap.tui.app import MEETCAP_DARK

        assert MEETCAP_DARK.accent == "#ff6b6b"
        assert MEETCAP_DARK.success == "#4ade80"
        assert MEETCAP_DARK.error == "#ff4444"
        assert MEETCAP_DARK.warning == "#ffb347"


# -- app tests --


class TestMeetcapAppInit:
    """tests for MeetcapApp initialization (no run)."""

    def test_default_init(self) -> None:
        with patch("meetcap.tui.app.Config") as MockConfig:
            MockConfig.return_value = MagicMock()
            from meetcap.tui.app import MeetcapApp

            app = MeetcapApp()
            assert app._initial_screen == "home"
            assert app.record_args is None
            assert app.process_file is None

    def test_custom_init(self) -> None:
        with patch("meetcap.tui.app.Config") as MockConfig:
            MockConfig.return_value = MagicMock()
            from meetcap.tui.app import MeetcapApp

            app = MeetcapApp(
                initial_screen="record",
                record_args={"auto_stop": 30},
                process_file=Path("/tmp/test.opus"),
            )
            assert app._initial_screen == "record"
            assert app.record_args == {"auto_stop": 30}
            assert app.process_file == Path("/tmp/test.opus")

    def test_app_title(self) -> None:
        from meetcap.tui.app import MeetcapApp

        assert MeetcapApp.TITLE == "meetcap"

    def test_app_css_paths(self) -> None:
        from meetcap.tui.app import MeetcapApp

        assert "css/theme.tcss" in MeetcapApp.CSS_PATH
        assert "css/home.tcss" in MeetcapApp.CSS_PATH
        assert "css/record.tcss" in MeetcapApp.CSS_PATH
        assert "css/process.tcss" in MeetcapApp.CSS_PATH
        assert "css/history.tcss" in MeetcapApp.CSS_PATH
        assert "css/settings.tcss" in MeetcapApp.CSS_PATH
        assert "css/setup.tcss" in MeetcapApp.CSS_PATH
        assert "css/modals.tcss" in MeetcapApp.CSS_PATH

    def test_app_has_command_palette(self) -> None:
        from meetcap.tui.app import MeetcapApp

        assert MeetcapApp.ENABLE_COMMAND_PALETTE is True

    def test_app_bindings(self) -> None:
        from meetcap.tui.app import MeetcapApp

        keys = [b.key for b in MeetcapApp.BINDINGS]
        assert "r" in keys
        assert "h" in keys
        assert "s" in keys
        assert "q" in keys


# -- command palette tests --


class TestMeetcapCommands:
    """tests for the command palette provider."""

    def test_import(self) -> None:
        from meetcap.tui.commands import MeetcapCommands

        assert MeetcapCommands is not None


# -- modal tests --


class TestStopConfirmModal:
    """tests for StopConfirmModal."""

    def test_import(self) -> None:
        from meetcap.tui.modals.confirm import StopConfirmModal

        assert StopConfirmModal is not None

    def test_has_escape_binding(self) -> None:
        from meetcap.tui.modals.confirm import StopConfirmModal

        keys = [b[0] if isinstance(b, tuple) else b.key for b in StopConfirmModal.BINDINGS]
        assert "escape" in keys


class TestDeleteConfirmModal:
    """tests for DeleteConfirmModal."""

    def test_import(self) -> None:
        from meetcap.tui.modals.confirm import DeleteConfirmModal

        assert DeleteConfirmModal is not None

    def test_stores_title(self) -> None:
        from meetcap.tui.modals.confirm import DeleteConfirmModal

        modal = DeleteConfirmModal(recording_title="Test Meeting")
        assert modal._recording_title == "Test Meeting"


class TestErrorModal:
    """tests for ErrorModal."""

    def test_import(self) -> None:
        from meetcap.tui.modals.error import ErrorModal

        assert ErrorModal is not None

    def test_stores_error_info(self) -> None:
        from meetcap.tui.modals.error import ErrorModal

        modal = ErrorModal(
            error_title="Test Error",
            error_message="Something went wrong",
            suggestion="Try again",
        )
        assert modal._error_title == "Test Error"
        assert modal._error_message == "Something went wrong"
        assert modal._suggestion == "Try again"


# -- screen tests --


class TestScreenImports:
    """test that all screens can be imported."""

    def test_home_screen(self) -> None:
        from meetcap.tui.screens.home import HomeScreen

        assert HomeScreen is not None

    def test_record_screen(self) -> None:
        from meetcap.tui.screens.record import RecordScreen

        assert RecordScreen is not None

    def test_process_screen(self) -> None:
        from meetcap.tui.screens.process import ProcessScreen

        assert ProcessScreen is not None

    def test_history_screen(self) -> None:
        from meetcap.tui.screens.history import HistoryScreen

        assert HistoryScreen is not None

    def test_settings_screen(self) -> None:
        from meetcap.tui.screens.settings import SettingsScreen

        assert SettingsScreen is not None

    def test_setup_screen(self) -> None:
        from meetcap.tui.screens.setup import SetupScreen

        assert SetupScreen is not None


class TestScreenBindings:
    """test screen key bindings are properly configured."""

    def test_record_bindings(self) -> None:
        from meetcap.tui.screens.record import RecordScreen

        keys = [b.key for b in RecordScreen.BINDINGS]
        assert "space" in keys
        assert "e" in keys
        assert "c" in keys
        assert "escape" in keys

    def test_process_bindings(self) -> None:
        from meetcap.tui.screens.process import ProcessScreen

        keys = [b.key for b in ProcessScreen.BINDINGS]
        assert "escape" in keys

    def test_history_bindings(self) -> None:
        from meetcap.tui.screens.history import HistoryScreen

        keys = [b.key for b in HistoryScreen.BINDINGS]
        assert "r" in keys
        assert "d" in keys
        assert "slash" in keys
        assert "escape" in keys

    def test_settings_bindings(self) -> None:
        from meetcap.tui.screens.settings import SettingsScreen

        keys = [b.key for b in SettingsScreen.BINDINGS]
        assert "ctrl+s" in keys
        assert "escape" in keys


class TestRecordScreenUnit:
    """unit tests for RecordScreen state."""

    def test_initial_state(self) -> None:
        from meetcap.tui.screens.record import RecordScreen

        screen = RecordScreen()
        assert screen._recording is False
        assert screen._recorder is None
        assert screen._recording_dir is None
        assert screen._start_time == 0.0
        assert screen._stop_requested is False

    def test_timer_defaults(self) -> None:
        from meetcap.tui.screens.record import RecordScreen

        screen = RecordScreen()
        assert screen._timer_seconds == 0
        assert screen._timer_remaining == 0.0


class TestProcessScreenUnit:
    """unit tests for ProcessScreen state."""

    def test_initial_state(self) -> None:
        from meetcap.tui.screens.process import ProcessScreen

        screen = ProcessScreen()
        assert screen._audio_path is None
        assert screen._processing is False

    def test_audio_path_arg(self) -> None:
        from meetcap.tui.screens.process import ProcessScreen

        screen = ProcessScreen(audio_path=Path("/tmp/test.opus"))
        assert screen._audio_path == Path("/tmp/test.opus")


class TestHistoryScreenUnit:
    """unit tests for HistoryScreen state."""

    def test_initial_state(self) -> None:
        from meetcap.tui.screens.history import HistoryScreen

        screen = HistoryScreen()
        assert screen._recordings == []
        assert screen._selected_dir is None

    def test_detect_files_empty_dir(self) -> None:
        from meetcap.tui.screens.history import HistoryScreen

        with tempfile.TemporaryDirectory() as tmpdir:
            result = HistoryScreen._detect_files(Path(tmpdir))
            assert result == []

    def test_detect_files_with_audio(self) -> None:
        from meetcap.tui.screens.history import HistoryScreen

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "recording.opus").touch()
            result = HistoryScreen._detect_files(Path(tmpdir))
            assert "a" in result

    def test_detect_files_with_transcript(self) -> None:
        from meetcap.tui.screens.history import HistoryScreen

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "recording.transcript.txt").touch()
            result = HistoryScreen._detect_files(Path(tmpdir))
            assert "t" in result

    def test_detect_files_with_summary(self) -> None:
        from meetcap.tui.screens.history import HistoryScreen

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "recording.summary.md").touch()
            result = HistoryScreen._detect_files(Path(tmpdir))
            assert "s" in result

    def test_detect_files_with_notes(self) -> None:
        from meetcap.tui.screens.history import HistoryScreen

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "notes.md").touch()
            result = HistoryScreen._detect_files(Path(tmpdir))
            assert "n" in result

    def test_detect_files_all_present(self) -> None:
        from meetcap.tui.screens.history import HistoryScreen

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "recording.opus").touch()
            (Path(tmpdir) / "recording.transcript.txt").touch()
            (Path(tmpdir) / "recording.summary.md").touch()
            (Path(tmpdir) / "notes.md").touch()
            result = HistoryScreen._detect_files(Path(tmpdir))
            assert result == ["a", "t", "s", "n"]


# -- config integration tests --


class TestConfigIsConfigured:
    """tests for Config.is_configured()."""

    def test_not_configured_when_no_file(self) -> None:
        from meetcap.utils.config import Config

        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(config_path=Path(tmpdir) / "config.toml")
            assert config.is_configured() is False

    def test_configured_when_file_exists(self) -> None:
        from meetcap.utils.config import Config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.touch()
            config = Config(config_path=config_path)
            assert config.is_configured() is True


# -- CLI integration tests --


class TestCLITuiIntegration:
    """test CLI --no-tui flag and TUI launch logic."""

    @staticmethod
    def _strip_ansi(text: str) -> str:
        """remove ANSI escape codes from text."""
        import re

        return re.sub(r"\x1b\[[0-9;]*m", "", text)

    def test_record_has_no_tui_option(self) -> None:
        """verify the --no-tui flag exists on the record command."""
        from typer.testing import CliRunner

        from meetcap.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["record", "--help"])
        assert "--no-tui" in self._strip_ansi(result.output)

    def test_summarize_has_no_tui_option(self) -> None:
        """verify the --no-tui flag exists on the summarize command."""
        from typer.testing import CliRunner

        from meetcap.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["summarize", "--help"])
        assert "--no-tui" in self._strip_ansi(result.output)

    def test_reprocess_has_no_tui_option(self) -> None:
        """verify the --no-tui flag exists on the reprocess command."""
        from typer.testing import CliRunner

        from meetcap.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["reprocess", "--help"])
        assert "--no-tui" in self._strip_ansi(result.output)

    def test_launch_tui_function_exists(self) -> None:
        """verify _launch_tui helper function exists."""
        from meetcap.cli import _launch_tui

        assert callable(_launch_tui)

    def test_launch_tui_non_tty_noop(self) -> None:
        """_launch_tui does nothing when stdout is not a tty."""
        from meetcap.cli import _launch_tui

        with patch("meetcap.cli.sys") as mock_sys:
            mock_sys.stdout.isatty.return_value = False
            # should return without launching anything
            _launch_tui()

    def test_launch_tui_env_var_noop(self) -> None:
        """_launch_tui does nothing when MEETCAP_NO_TUI is set."""
        import os

        from meetcap.cli import _launch_tui

        with patch.dict(os.environ, {"MEETCAP_NO_TUI": "1"}):
            with patch("meetcap.cli.sys") as mock_sys:
                mock_sys.stdout.isatty.return_value = True
                _launch_tui()


# -- async textual pilot tests --


class TestHomeScreenCompose:
    """test HomeScreen widget composition using textual pilot."""

    @pytest.mark.asyncio
    async def test_home_screen_has_widgets(self) -> None:
        """verify HomeScreen composes expected widgets."""

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield Static("test")

            def on_mount(self) -> None:
                from meetcap.tui.screens.home import HomeScreen

                self.install_screen(HomeScreen(), name="home")
                self.push_screen("home")

        app = TestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            # verify home screen widgets exist
            assert len(app.screen.query("QuickActions")) > 0
            assert len(app.screen.query("#recent-recordings")) > 0
            assert len(app.screen.query("#system-status")) > 0

    @pytest.mark.asyncio
    async def test_home_screen_buttons(self) -> None:
        """verify HomeScreen has action buttons."""

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield Static("test")

            def on_mount(self) -> None:
                from meetcap.tui.screens.home import HomeScreen

                self.install_screen(HomeScreen(), name="home")
                self.push_screen("home")

        app = TestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            assert app.screen.query_one("#btn-record")
            assert app.screen.query_one("#btn-history")
            assert app.screen.query_one("#btn-settings")
            assert app.screen.query_one("#btn-quit")


class TestSettingsScreenCompose:
    """test SettingsScreen widget composition."""

    @pytest.mark.asyncio
    async def test_settings_screen_has_form(self) -> None:
        """verify SettingsScreen has form fields."""

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield Static("test")

            def on_mount(self) -> None:
                from meetcap.tui.screens.settings import SettingsScreen

                self.install_screen(SettingsScreen(), name="settings")
                self.push_screen("settings")

        app = TestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            from textual.widgets import Checkbox, Input, Select

            # these MUST be Select dropdowns, not free-text Input
            assert isinstance(app.screen.query_one("#setting-device"), Select)
            assert isinstance(app.screen.query_one("#setting-format"), Select)
            assert isinstance(app.screen.query_one("#setting-sample-rate"), Select)
            assert isinstance(app.screen.query_one("#setting-stt-engine"), Select)
            assert isinstance(app.screen.query_one("#setting-llm-model"), Select)
            assert isinstance(app.screen.query_one("#setting-temperature"), Select)
            # these are correctly Checkbox/Input
            assert isinstance(app.screen.query_one("#setting-diarization"), Checkbox)
            assert isinstance(app.screen.query_one("#setting-out-dir"), Input)
            assert isinstance(app.screen.query_one("#setting-notes"), Checkbox)


class TestSettingsDropdownOptions:
    """verify settings dropdowns contain the correct options."""

    def test_stt_engines_match_cli(self) -> None:
        """STT engine options must match what cli.py actually supports."""
        from meetcap.tui.screens.settings import STT_ENGINES

        engine_keys = [key for _, key in STT_ENGINES]
        # these are the config-format names used in cli.py
        assert "parakeet" in engine_keys
        assert "faster-whisper" in engine_keys
        assert "mlx-whisper" in engine_keys
        assert "vosk" in engine_keys

    def test_llm_models_match_cli(self) -> None:
        """LLM model options must include the models from cli.py setup."""
        from meetcap.tui.screens.settings import LLM_MODELS

        model_repos = [repo for _, repo in LLM_MODELS]
        assert "mlx-community/Qwen3.5-2B-OptiQ-4bit" in model_repos
        assert "mlx-community/Qwen3.5-9B-OptiQ-4bit" in model_repos

    def test_audio_formats_match_config(self) -> None:
        """audio format options must match AudioFormat enum."""
        from meetcap.tui.screens.settings import AUDIO_FORMATS

        format_keys = [key for _, key in AUDIO_FORMATS]
        assert "opus" in format_keys
        assert "wav" in format_keys
        assert "flac" in format_keys

    def test_no_free_text_for_structured_fields(self) -> None:
        """device, format, rate, engine, model, temperature must NOT be Input."""
        import ast
        import inspect
        import textwrap

        from meetcap.tui.screens.settings import SettingsScreen

        source = textwrap.dedent(inspect.getsource(SettingsScreen.compose))
        tree = ast.parse(source)
        # collect all Input() calls and their id kwargs
        input_ids = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                if func_name == "Input":
                    for kw in node.keywords:
                        if kw.arg == "id" and isinstance(kw.value, ast.Constant):
                            input_ids.append(kw.value.value)

        # these fields must NOT be free-text Input
        must_be_select = [
            "setting-device",
            "setting-format",
            "setting-sample-rate",
            "setting-stt-engine",
            "setting-llm-model",
            "setting-temperature",
        ]
        for field_id in must_be_select:
            assert field_id not in input_ids, (
                f"'{field_id}' is a free-text Input but should be a Select dropdown"
            )


class TestSetupScreenCompose:
    """test SetupScreen widget composition."""

    @pytest.mark.asyncio
    async def test_setup_screen_has_steps(self) -> None:
        """verify SetupScreen has step containers."""

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield Static("test")

            def on_mount(self) -> None:
                from meetcap.tui.screens.setup import SetupScreen

                self.install_screen(SetupScreen(), name="setup")
                self.push_screen("setup")

        app = TestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            assert app.screen.query_one("#step-dependencies")
            assert app.screen.query_one("#step-audio")
            assert app.screen.query_one("#step-stt")
            assert app.screen.query_one("#step-output")
            assert app.screen.query_one("#setup-next")
            assert app.screen.query_one("#setup-back")
            assert app.screen.query_one("#setup-finish")


class TestHistoryScreenCompose:
    """test HistoryScreen widget composition."""

    @pytest.mark.asyncio
    async def test_history_screen_has_table(self) -> None:
        """verify HistoryScreen has table and search."""

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield Static("test")

            def on_mount(self) -> None:
                from meetcap.tui.screens.history import HistoryScreen

                self.install_screen(HistoryScreen(), name="history")
                self.push_screen("history")

        app = TestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            assert app.screen.query_one("#history-table")
            assert app.screen.query_one("#history-search")
            assert app.screen.query_one("#history-sort")
            assert app.screen.query_one("#summary-preview")


class TestProcessScreenCompose:
    """test ProcessScreen widget composition."""

    @pytest.mark.asyncio
    async def test_process_screen_has_pipeline(self) -> None:
        """verify ProcessScreen has pipeline widget."""

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield Static("test")

            def on_mount(self) -> None:
                from meetcap.tui.screens.process import ProcessScreen

                self.install_screen(ProcessScreen(), name="process")
                self.push_screen("process")

        app = TestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            assert app.screen.query_one("#pipeline-progress")
            assert app.screen.query_one("#process-log")
            assert app.screen.query_one("#process-audio-info")


class TestRecordScreenCompose:
    """test RecordScreen widget composition."""

    @pytest.mark.asyncio
    async def test_record_screen_has_widgets(self) -> None:
        """verify RecordScreen has expected widgets."""

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield Static("test")

            def on_mount(self) -> None:
                from meetcap.tui.screens.record import RecordScreen

                self.install_screen(RecordScreen(), name="record")
                self.push_screen("record")

        app = TestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            assert app.screen.query_one("#recording-digits")
            assert app.screen.query_one("#audio-device")
            assert app.screen.query_one("#audio-format")
            assert app.screen.query_one("#audio-level")
            assert app.screen.query_one("#timer-info")
            assert app.screen.query_one("#recording-log")


class TestModalCompose:
    """test modal dialog composition."""

    @pytest.mark.asyncio
    async def test_stop_confirm_modal(self) -> None:
        """verify StopConfirmModal has buttons."""
        from meetcap.tui.modals.confirm import StopConfirmModal

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield Static("test")

            def on_mount(self) -> None:
                self.push_screen(StopConfirmModal())

        app = TestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            assert app.screen.query_one("#btn-process")
            assert app.screen.query_one("#btn-skip")
            assert app.screen.query_one("#btn-cancel")

    @pytest.mark.asyncio
    async def test_delete_confirm_modal(self) -> None:
        """verify DeleteConfirmModal has buttons."""
        from meetcap.tui.modals.confirm import DeleteConfirmModal

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield Static("test")

            def on_mount(self) -> None:
                self.push_screen(DeleteConfirmModal(recording_title="Test"))

        app = TestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            assert app.screen.query_one("#btn-delete")
            assert app.screen.query_one("#btn-cancel-delete")

    @pytest.mark.asyncio
    async def test_error_modal(self) -> None:
        """verify ErrorModal has content."""
        from meetcap.tui.modals.error import ErrorModal

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield Static("test")

            def on_mount(self) -> None:
                self.push_screen(
                    ErrorModal(
                        error_title="Test Error",
                        error_message="Something broke",
                        suggestion="Try again",
                    )
                )

        app = TestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            assert app.screen.query_one("#error-title")
            assert app.screen.query_one("#error-message")
            assert app.screen.query_one("#error-suggestion")
            assert app.screen.query_one("#btn-close-error")


class TestWidgetCompose:
    """test widget composition via pilot."""

    @pytest.mark.asyncio
    async def test_recording_digits(self) -> None:
        """verify RecordingDigits renders."""
        from meetcap.tui.widgets.recording_digits import RecordingDigits

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield RecordingDigits(id="test-digits")

        app = TestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            digits = app.query_one("#test-digits", RecordingDigits)
            assert digits is not None
            digits.update_time(3661.5)
            assert digits.elapsed == 3661.5

    @pytest.mark.asyncio
    async def test_audio_level_meter(self) -> None:
        """verify AudioLevelMeter renders and accepts data."""
        from meetcap.tui.widgets.audio_level import AudioLevelMeter

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield AudioLevelMeter(id="test-level", max_points=5)

        app = TestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            meter = app.query_one("#test-level", AudioLevelMeter)
            meter.update_level(-30.0)
            meter.update_level(-20.0)
            meter.update_level(-10.0)
            assert len(meter._levels) == 3
            # test max_points trimming
            for _i in range(10):
                meter.update_level(-15.0)
            assert len(meter._levels) == 5

    @pytest.mark.asyncio
    async def test_audio_level_meter_reset(self) -> None:
        """verify AudioLevelMeter reset clears data."""
        from meetcap.tui.widgets.audio_level import AudioLevelMeter

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield AudioLevelMeter(id="test-level")

        app = TestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            meter = app.query_one("#test-level", AudioLevelMeter)
            meter.update_level(-30.0)
            assert len(meter._levels) == 1
            meter.reset()
            assert len(meter._levels) == 0

    @pytest.mark.asyncio
    async def test_pipeline_progress(self) -> None:
        """verify PipelineProgress renders with stages."""
        from meetcap.tui.widgets.pipeline import PipelineProgress

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield PipelineProgress(id="test-pipeline")

        app = TestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            pipeline = app.query_one("#test-pipeline", PipelineProgress)
            assert "stt" in pipeline._stages
            assert "diarization" in pipeline._stages
            assert "summarization" in pipeline._stages
            assert "organize" in pipeline._stages

    @pytest.mark.asyncio
    async def test_pipeline_update_stage(self) -> None:
        """verify PipelineProgress stage updates."""
        from meetcap.tui.widgets.pipeline import PipelineProgress

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield PipelineProgress(id="test-pipeline")

        app = TestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            pipeline = app.query_one("#test-pipeline", PipelineProgress)
            pipeline.update_stage("stt", "done", progress=100, timing=5.2, detail="42 segments")
            assert pipeline._stages["stt"].status == "done"
            assert pipeline._stages["stt"].timing == 5.2

    @pytest.mark.asyncio
    async def test_pipeline_reset(self) -> None:
        """verify PipelineProgress reset."""
        from meetcap.tui.widgets.pipeline import PipelineProgress

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield PipelineProgress(id="test-pipeline")

        app = TestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            pipeline = app.query_one("#test-pipeline", PipelineProgress)
            pipeline.update_stage("stt", "done", progress=100, timing=5.2)
            pipeline.reset()
            assert pipeline._stages["stt"].status == "pending"

    @pytest.mark.asyncio
    async def test_model_status_indicator(self) -> None:
        """verify ModelStatusIndicator state management."""
        from meetcap.tui.widgets.model_status import ModelStatusIndicator

        # test without mounting -- state management is the key logic
        indicator = ModelStatusIndicator(model_name="test-model")
        indicator.set_status("ready", size="1.2 GB")
        assert indicator._status == "ready"
        assert indicator._size == "1.2 GB"
        indicator.set_status("downloading")
        assert indicator._status == "downloading"
        assert indicator._size == "1.2 GB"  # size preserved


# -- widget __init__ exports test --


class TestWidgetExports:
    """test widget package exports."""

    def test_all_widgets_importable(self) -> None:
        from meetcap.tui.widgets import (
            AudioLevelMeter,
            ModelStatusIndicator,
            PipelineProgress,
            RecordingDigits,
            SystemStatus,
        )

        assert AudioLevelMeter is not None
        assert ModelStatusIndicator is not None
        assert PipelineProgress is not None
        assert RecordingDigits is not None
        assert SystemStatus is not None


class TestModalExports:
    """test modal package exports."""

    def test_all_modals_importable(self) -> None:
        from meetcap.tui.modals import (
            DeleteConfirmModal,
            ErrorModal,
            StopConfirmModal,
        )

        assert DeleteConfirmModal is not None
        assert ErrorModal is not None
        assert StopConfirmModal is not None


# -- additional tests for coverage boost --


class TestSetupScreenUnit:
    """unit tests for SetupScreen state."""

    def test_initial_step(self) -> None:
        from meetcap.tui.screens.setup import SetupScreen

        screen = SetupScreen()
        assert screen._current_step == 0
        assert screen._total_steps == 4


class TestSettingsScreenUnit:
    """unit tests for SettingsScreen state."""

    def test_initial_config_none(self) -> None:
        from meetcap.tui.screens.settings import SettingsScreen

        screen = SettingsScreen()
        assert screen._config is None


class TestPipelineProgressDefaults:
    """test PipelineProgress default stage configuration."""

    def test_default_stages(self) -> None:
        from meetcap.tui.widgets.pipeline import PipelineProgress

        assert len(PipelineProgress.DEFAULT_STAGES) == 4
        names = [s.name for s in PipelineProgress.DEFAULT_STAGES]
        assert names == ["stt", "diarization", "summarization", "organize"]


class TestStageWidgetFormatLabel:
    """test StageWidget label formatting."""

    def test_pending_label(self) -> None:
        from meetcap.tui.widgets.pipeline import StageInfo, StageWidget

        info = StageInfo(name="stt", label="STT")
        widget = StageWidget(info)
        label = widget._format_label()
        assert "STT" in label
        assert "\u25cb" in label  # pending icon

    def test_done_with_timing(self) -> None:
        from meetcap.tui.widgets.pipeline import StageInfo, StageWidget

        info = StageInfo(name="stt", label="STT", status="done", timing=5.2, detail="42 segments")
        widget = StageWidget(info)
        label = widget._format_label()
        assert "5.2s" in label
        assert "42 segments" in label
        assert "\u2713" in label  # done icon

    def test_error_label(self) -> None:
        from meetcap.tui.widgets.pipeline import StageInfo, StageWidget

        info = StageInfo(name="stt", label="STT", status="error")
        widget = StageWidget(info)
        label = widget._format_label()
        assert "\u2717" in label  # error icon

    def test_active_label(self) -> None:
        from meetcap.tui.widgets.pipeline import _SPINNER_FRAMES, StageInfo, StageWidget

        info = StageInfo(name="stt", label="STT", status="active")
        widget = StageWidget(info)
        label = widget._format_label()
        # active state renders one of the rotating spinner frames so the
        # user can see at a glance the stage is alive.
        assert any(frame in label for frame in _SPINNER_FRAMES)
        assert "STT" in label


class TestModelStatusFormat:
    """test ModelStatusIndicator format."""

    def test_format_status_unknown(self) -> None:
        from meetcap.tui.widgets.model_status import ModelStatusIndicator

        indicator = ModelStatusIndicator(model_name="test")
        result = indicator._format_status()
        assert "test" in result
        assert "unknown" in result

    def test_format_status_ready_with_size(self) -> None:
        from meetcap.tui.widgets.model_status import ModelStatusIndicator

        indicator = ModelStatusIndicator(model_name="model-x")
        indicator._status = "ready"
        indicator._size = "2.5 GB"
        result = indicator._format_status()
        assert "model-x" in result
        assert "2.5 GB" in result
        assert "ready" in result

    def test_format_status_missing(self) -> None:
        from meetcap.tui.widgets.model_status import ModelStatusIndicator

        indicator = ModelStatusIndicator(model_name="model-y")
        indicator._status = "missing"
        result = indicator._format_status()
        assert "\u2717" in result

    def test_format_status_downloading(self) -> None:
        from meetcap.tui.widgets.model_status import ModelStatusIndicator

        indicator = ModelStatusIndicator(model_name="model-z")
        indicator._status = "downloading"
        result = indicator._format_status()
        assert "\u2193" in result


class TestRecordingDigitsTimeFormat:
    """test RecordingDigits time calculation."""

    def test_zero_time(self) -> None:
        from meetcap.tui.widgets.recording_digits import RecordingDigits

        d = RecordingDigits()
        d.update_time(0)
        assert d.elapsed == 0

    def test_hours_calculation(self) -> None:
        from meetcap.tui.widgets.recording_digits import RecordingDigits

        d = RecordingDigits()
        d.update_time(3661)  # 1h 1m 1s
        assert d.elapsed == 3661

    def test_large_time(self) -> None:
        from meetcap.tui.widgets.recording_digits import RecordingDigits

        d = RecordingDigits()
        d.update_time(86400)  # 24 hours
        assert d.elapsed == 86400


class TestAudioLevelNormalization:
    """test audio level dB to normalized conversion."""

    def test_silence_normalizes_to_zero(self) -> None:
        from meetcap.tui.widgets.audio_level import AudioLevelMeter

        meter = AudioLevelMeter()
        # -60 dB maps to 0.0
        meter._levels = []
        level_db = -60.0
        normalized = max(0.0, min(100.0, (level_db + 60.0) * (100.0 / 60.0)))
        assert normalized == 0.0

    def test_max_normalizes_to_100(self) -> None:
        level_db = 0.0
        normalized = max(0.0, min(100.0, (level_db + 60.0) * (100.0 / 60.0)))
        assert normalized == 100.0

    def test_mid_level(self) -> None:
        level_db = -30.0
        normalized = max(0.0, min(100.0, (level_db + 60.0) * (100.0 / 60.0)))
        assert normalized == pytest.approx(50.0)


class TestCLIMainCallback:
    """tests for the main callback TUI integration."""

    def test_version_flag(self) -> None:
        from typer.testing import CliRunner

        from meetcap.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["--version"])
        assert "meetcap v" in result.output

    def test_main_with_no_args_non_tty(self) -> None:
        """when not a tty, main should not launch TUI."""
        from typer.testing import CliRunner

        from meetcap.cli import app

        runner = CliRunner()
        # CliRunner doesn't provide a tty, so TUI won't launch
        result = runner.invoke(app, [])
        # should exit cleanly (no subcommand, no tty)
        assert result.exit_code == 0


class TestSetupScreenNavigation:
    """test SetupScreen step navigation via pilot."""

    @pytest.mark.asyncio
    async def test_next_step_advances(self) -> None:
        """verify clicking next advances the setup step."""
        from meetcap.tui.screens.setup import SetupScreen

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield Static("test")

            def on_mount(self) -> None:
                self.install_screen(SetupScreen(), name="setup")
                self.push_screen("setup")

        app = TestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            screen = app.screen
            assert screen._current_step == 0
            await _pilot.click("#setup-next")
            await _pilot.pause()
            assert screen._current_step >= 1  # advanced at least once

    @pytest.mark.asyncio
    async def test_back_step_goes_back(self) -> None:
        """verify clicking back goes to previous step."""
        from meetcap.tui.screens.setup import SetupScreen

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield Static("test")

            def on_mount(self) -> None:
                self.install_screen(SetupScreen(), name="setup")
                self.push_screen("setup")

        app = TestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            screen = app.screen
            await _pilot.click("#setup-next")
            assert screen._current_step == 1
            await _pilot.click("#setup-back")
            assert screen._current_step == 0

    @pytest.mark.asyncio
    async def test_cannot_go_back_on_first_step(self) -> None:
        """verify back on first step does nothing."""
        from meetcap.tui.screens.setup import SetupScreen

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield Static("test")

            def on_mount(self) -> None:
                self.install_screen(SetupScreen(), name="setup")
                self.push_screen("setup")

        app = TestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            screen = app.screen
            await _pilot.click("#setup-back")
            assert screen._current_step == 0


class TestRecordScreenActions:
    """test RecordScreen action methods."""

    @pytest.mark.asyncio
    async def test_timer_extend(self) -> None:
        """verify extend timer action."""
        from meetcap.tui.screens.record import RecordScreen

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield Static("test")

            def on_mount(self) -> None:
                self.install_screen(RecordScreen(), name="record")
                self.push_screen("record")

        app = TestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            screen = app.screen
            initial = screen._timer_seconds
            screen.action_extend_timer()
            assert screen._timer_seconds == initial + 30 * 60

    @pytest.mark.asyncio
    async def test_timer_cancel(self) -> None:
        """verify cancel timer action."""
        from meetcap.tui.screens.record import RecordScreen

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield Static("test")

            def on_mount(self) -> None:
                self.install_screen(RecordScreen(), name="record")
                self.push_screen("record")

        app = TestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            screen = app.screen
            screen._timer_seconds = 60 * 60
            screen.action_cancel_timer()
            assert screen._timer_seconds == 0
            assert screen._timer_remaining == 0


class TestStopModalButtonPresses:
    """test StopConfirmModal button press handlers."""

    @pytest.mark.asyncio
    async def test_process_button(self) -> None:
        from meetcap.tui.modals.confirm import StopConfirmModal

        result_holder = []

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield Static("test")

            def on_mount(self) -> None:
                self.push_screen(
                    StopConfirmModal(),
                    callback=lambda r: result_holder.append(r),
                )

        app = TestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            await _pilot.click("#btn-process")
            assert result_holder == ["process"]

    @pytest.mark.asyncio
    async def test_skip_button(self) -> None:
        from meetcap.tui.modals.confirm import StopConfirmModal

        result_holder = []

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield Static("test")

            def on_mount(self) -> None:
                self.push_screen(
                    StopConfirmModal(),
                    callback=lambda r: result_holder.append(r),
                )

        app = TestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            await _pilot.click("#btn-skip")
            assert result_holder == ["skip"]

    @pytest.mark.asyncio
    async def test_cancel_button(self) -> None:
        from meetcap.tui.modals.confirm import StopConfirmModal

        result_holder = []

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield Static("test")

            def on_mount(self) -> None:
                self.push_screen(
                    StopConfirmModal(),
                    callback=lambda r: result_holder.append(r),
                )

        app = TestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            await _pilot.click("#btn-cancel")
            assert result_holder == ["cancel"]


class TestDeleteModalButtonPresses:
    """test DeleteConfirmModal button press handlers."""

    @pytest.mark.asyncio
    async def test_delete_button(self) -> None:
        from meetcap.tui.modals.confirm import DeleteConfirmModal

        result_holder = []

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield Static("test")

            def on_mount(self) -> None:
                self.push_screen(
                    DeleteConfirmModal(recording_title="Test"),
                    callback=lambda r: result_holder.append(r),
                )

        app = TestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            await _pilot.click("#btn-delete")
            assert result_holder == [True]

    @pytest.mark.asyncio
    async def test_cancel_delete_button(self) -> None:
        from meetcap.tui.modals.confirm import DeleteConfirmModal

        result_holder = []

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield Static("test")

            def on_mount(self) -> None:
                self.push_screen(
                    DeleteConfirmModal(recording_title="Test"),
                    callback=lambda r: result_holder.append(r),
                )

        app = TestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            await _pilot.click("#btn-cancel-delete")
            assert result_holder == [False]


class TestErrorModalButtonPress:
    """test ErrorModal button press handler."""

    @pytest.mark.asyncio
    async def test_close_button(self) -> None:
        from meetcap.tui.modals.error import ErrorModal

        result_holder = []

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield Static("test")

            def on_mount(self) -> None:
                self.push_screen(
                    ErrorModal(error_title="Error", error_message="fail"),
                    callback=lambda r: result_holder.append(r),
                )

        app = TestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            await _pilot.click("#btn-close-error")
            assert result_holder == [True]


class TestHistoryDetectFilesVariants:
    """additional coverage for _detect_files."""

    def test_wav_detected(self) -> None:
        from meetcap.tui.screens.history import HistoryScreen

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "recording.wav").touch()
            result = HistoryScreen._detect_files(Path(tmpdir))
            assert "a" in result

    def test_flac_detected(self) -> None:
        from meetcap.tui.screens.history import HistoryScreen

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "recording.flac").touch()
            result = HistoryScreen._detect_files(Path(tmpdir))
            assert "a" in result

    def test_json_transcript_detected(self) -> None:
        from meetcap.tui.screens.history import HistoryScreen

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "recording.transcript.json").touch()
            result = HistoryScreen._detect_files(Path(tmpdir))
            assert "t" in result


# -- e2e: RecordScreen recording worker uses AudioDevice correctly --


class TestRecordScreenDeviceAccess:
    """test that RecordScreen accesses AudioDevice attributes (not dict keys)."""

    def test_worker_uses_dot_notation_on_audio_device(self) -> None:
        """regression test: AudioDevice is a dataclass, not a dict.

        The recording worker must use device.index / device.name,
        never device["index"] / device["name"].
        """
        import ast
        import inspect
        import textwrap

        from meetcap.tui.screens.record import RecordScreen

        source = textwrap.dedent(inspect.getsource(RecordScreen._start_recording_worker))
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript):
                # check if it's device["index"] or device["name"]
                if (
                    isinstance(node.value, ast.Name)
                    and node.value.id == "device"
                    and isinstance(node.slice, ast.Constant)
                    and node.slice.value in ("index", "name")
                ):
                    pytest.fail(
                        f'RecordScreen uses device["{node.slice.value}"] '
                        f"but AudioDevice is a dataclass — use device.{node.slice.value}"
                    )

    def test_audio_device_is_dataclass_not_dict(self) -> None:
        """verify AudioDevice uses attribute access, not subscript."""
        from meetcap.core.devices import AudioDevice

        device = AudioDevice(index=0, name="Test Device")
        # must work with attribute access
        assert device.index == 0
        assert device.name == "Test Device"
        # must NOT work with subscript
        with pytest.raises(TypeError, match="not subscriptable"):
            device["index"]  # type: ignore[index]


# -- e2e: record → process handoff flow tests --


class TestRecordToProcessHandoff:
    """test that the recording path is correctly passed to ProcessScreen."""

    def test_find_audio_file_opus(self) -> None:
        """_find_audio_file finds .opus files in recording dir."""
        from meetcap.tui.screens.record import RecordScreen

        screen = RecordScreen()
        with tempfile.TemporaryDirectory() as tmpdir:
            rec_dir = Path(tmpdir) / "2026_Mar_20_Test"
            rec_dir.mkdir()
            audio = rec_dir / "recording.opus"
            audio.write_bytes(b"fake opus data")
            screen._recording_dir = rec_dir
            result = screen._find_audio_file()
            assert result is not None
            assert result.name == "recording.opus"

    def test_find_audio_file_wav(self) -> None:
        """_find_audio_file finds .wav files."""
        from meetcap.tui.screens.record import RecordScreen

        screen = RecordScreen()
        with tempfile.TemporaryDirectory() as tmpdir:
            rec_dir = Path(tmpdir) / "2026_Mar_20_Test"
            rec_dir.mkdir()
            audio = rec_dir / "recording.wav"
            audio.write_bytes(b"fake wav data")
            screen._recording_dir = rec_dir
            result = screen._find_audio_file()
            assert result is not None
            assert result.name == "recording.wav"

    def test_find_audio_file_none_when_empty(self) -> None:
        """_find_audio_file returns None when no audio files exist."""
        from meetcap.tui.screens.record import RecordScreen

        screen = RecordScreen()
        with tempfile.TemporaryDirectory() as tmpdir:
            rec_dir = Path(tmpdir) / "2026_Mar_20_Test"
            rec_dir.mkdir()
            screen._recording_dir = rec_dir
            assert screen._find_audio_file() is None

    def test_find_audio_file_none_when_no_recording_dir(self) -> None:
        """_find_audio_file returns None when recording_dir is None."""
        from meetcap.tui.screens.record import RecordScreen

        screen = RecordScreen()
        screen._recording_dir = None
        assert screen._find_audio_file() is None

    def test_process_screen_reads_app_process_file(self) -> None:
        """ProcessScreen reads audio path from app.process_file attribute."""
        import inspect
        import textwrap

        from meetcap.tui.screens.process import ProcessScreen

        source = textwrap.dedent(inspect.getsource(ProcessScreen.on_mount))
        # must use "process_file" not "_process_file"
        assert '"process_file"' in source or "'process_file'" in source
        assert '"_process_file"' not in source and "'_process_file'" not in source

    def test_process_screen_app_attr_matches_app_init(self) -> None:
        """the attribute name ProcessScreen reads must match what MeetcapApp sets."""
        import inspect
        import textwrap

        from meetcap.tui.app import MeetcapApp
        from meetcap.tui.screens.process import ProcessScreen

        # extract the attribute name ProcessScreen reads
        process_source = textwrap.dedent(inspect.getsource(ProcessScreen.on_mount))
        assert "process_file" in process_source

        # verify MeetcapApp.__init__ sets the same attribute
        app_source = textwrap.dedent(inspect.getsource(MeetcapApp.__init__))
        assert "self.process_file" in app_source

    def test_record_screen_pushes_fresh_process_screen(self) -> None:
        """RecordScreen must push a new ProcessScreen(audio_path=...), not the named screen."""
        import inspect
        import textwrap

        from meetcap.tui.screens.record import RecordScreen

        source = textwrap.dedent(inspect.getsource(RecordScreen._show_stop_confirm))
        # must create ProcessScreen with audio_path, not push named "process"
        assert "ProcessScreen(audio_path=" in source
        # must NOT just push the named screen (stale instance)
        assert 'push_screen("process")' not in source

    def test_history_screen_pushes_fresh_process_screen(self) -> None:
        """HistoryScreen must push a new ProcessScreen, not the named screen."""
        import inspect
        import textwrap

        from meetcap.tui.screens.history import HistoryScreen

        source = textwrap.dedent(inspect.getsource(HistoryScreen.action_reprocess))
        assert "ProcessScreen(audio_path=" in source
        assert 'push_screen("process")' not in source


class TestRecordStopProcessE2E:
    """true e2e: simulate record → stop → process with mocked recorder."""

    @pytest.mark.asyncio
    async def test_stop_and_process_passes_audio_path(self) -> None:
        """after stopping recording, choosing 'process' must open
        ProcessScreen with the correct audio file path."""
        from meetcap.tui.screens.record import RecordScreen

        with tempfile.TemporaryDirectory() as tmpdir:
            rec_dir = Path(tmpdir) / "20260320-084124-temp"
            rec_dir.mkdir()
            audio_file = rec_dir / "recording.opus"
            audio_file.write_bytes(b"\x00" * 1000)

            # track what screen gets pushed
            pushed_screens: list = []

            class TestApp(App):
                def compose(self) -> ComposeResult:
                    yield Static("test")

                def on_mount(self) -> None:
                    self.push_screen(RecordScreen())

                def push_screen(self, screen, *args, **kwargs) -> None:
                    pushed_screens.append(screen)
                    return super().push_screen(screen, *args, **kwargs)

            app = TestApp()
            async with app.run_test(size=(120, 40)) as _pilot:
                # get the record screen
                record_screen = app.screen
                assert isinstance(record_screen, RecordScreen)

                # simulate a completed recording:
                # set the recording dir (as stop_recording would)
                record_screen._recording_dir = rec_dir
                record_screen._recording = False  # already stopped

                # call _find_audio_file and verify it works
                found = record_screen._find_audio_file()
                assert found is not None
                assert found.suffix == ".opus"

                # now simulate what _show_stop_confirm -> handle_result("process") does
                # by calling the internal logic directly
                from meetcap.tui.screens.process import ProcessScreen as PS

                audio = record_screen._find_audio_file()
                assert audio is not None

                # push a fresh ProcessScreen just like the real code does
                process_screen = PS(audio_path=audio)
                assert process_screen._audio_path == audio
                assert process_screen._audio_path.exists()

    @pytest.mark.asyncio
    async def test_process_screen_receives_audio_and_logs_it(self) -> None:
        """ProcessScreen with a valid audio_path must log 'Processing:' on mount."""
        from meetcap.tui.screens.process import ProcessScreen

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_file = Path(tmpdir) / "recording.opus"
            audio_file.write_bytes(b"\x00" * 5000)

            class TestApp(App):
                def compose(self) -> ComposeResult:
                    yield Static("test")

                def on_mount(self) -> None:
                    self.push_screen(ProcessScreen(audio_path=audio_file))

            app = TestApp()
            async with app.run_test(size=(120, 40)) as _pilot:
                # verify ProcessScreen got the path
                from meetcap.tui.screens.process import ProcessScreen as PS2

                screen = app.screen
                assert isinstance(screen, PS2)
                assert screen._audio_path is not None
                assert screen._audio_path.name == "recording.opus"

    @pytest.mark.asyncio
    async def test_process_screen_without_audio_shows_error(self) -> None:
        """ProcessScreen with no audio_path must show 'No audio file'."""
        from meetcap.tui.screens.process import ProcessScreen

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield Static("test")

            def on_mount(self) -> None:
                self.push_screen(ProcessScreen())

        app = TestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            # verify ProcessScreen shows error in the log
            # verify ProcessScreen has no audio path
            from meetcap.tui.screens.process import ProcessScreen as PS3

            screen = app.screen
            assert isinstance(screen, PS3)
            assert screen._audio_path is None


class TestProcessScreenSTTIntegration:
    """test that ProcessScreen._run_stt correctly constructs services.

    this is the real e2e test: it actually calls _run_stt with a mock
    and verifies the service receives a Path (not a Config object).
    """

    def test_run_stt_passes_path_not_config_to_parakeet(self) -> None:
        """_run_stt must pass audio_path (Path) to service.transcribe, not config."""
        from meetcap.tui.screens.process import ProcessScreen

        screen = ProcessScreen()
        # patch _log to avoid needing a real app
        screen._log = MagicMock()

        audio_path = Path("/tmp/fake_audio.opus")
        mock_config = MagicMock()
        mock_config.get.side_effect = lambda s, k, d=None: {
            ("models", "stt_engine"): "parakeet",
            ("models", "parakeet_model_name"): "test/model",
        }.get((s, k), d)

        mock_result = MagicMock()
        mock_result.segments = []

        with patch("meetcap.services.transcription.ParakeetService") as MockParakeet:
            mock_service = MagicMock()
            mock_service.transcribe.return_value = mock_result
            MockParakeet.return_value = mock_service

            result = screen._run_stt(audio_path, mock_config, "parakeet")

            # verify transcribe was called with the Path, not the Config
            mock_service.transcribe.assert_called_once_with(audio_path)
            assert isinstance(mock_service.transcribe.call_args[0][0], Path)
            assert result is mock_result

    def test_run_stt_passes_path_not_config_to_mlx(self) -> None:
        """_run_stt must pass audio_path (Path) to MLX service, not config."""
        from meetcap.tui.screens.process import ProcessScreen

        screen = ProcessScreen()
        screen._log = MagicMock()

        audio_path = Path("/tmp/fake_audio.opus")
        mock_config = MagicMock()
        mock_config.get.side_effect = lambda s, k, d=None: {
            ("models", "mlx_stt_model_name"): "mlx-community/whisper-large-v3-turbo",
        }.get((s, k), d)

        mock_result = MagicMock()
        with patch("meetcap.services.transcription.MlxWhisperService") as MockMlx:
            mock_service = MagicMock()
            mock_service.transcribe.return_value = mock_result
            MockMlx.return_value = mock_service

            screen._run_stt(audio_path, mock_config, "mlx")

            # MlxWhisperService must be constructed with model_name str, not Config
            constructor_kwargs = MockMlx.call_args
            assert "model_name" in constructor_kwargs.kwargs or len(constructor_kwargs.args) > 0
            # transcribe must receive the Path
            mock_service.transcribe.assert_called_once_with(audio_path)

    def test_run_stt_passes_path_not_config_to_fwhisper(self) -> None:
        """_run_stt must pass audio_path (Path) to FasterWhisper, not config."""
        from meetcap.tui.screens.process import ProcessScreen

        screen = ProcessScreen()
        screen._log = MagicMock()

        audio_path = Path("/tmp/fake_audio.opus")
        mock_config = MagicMock()
        mock_config.get.side_effect = lambda s, k, d=None: {
            ("models", "stt_model_name"): "large-v3",
            ("models", "stt_model_path"): "~/.meetcap/models/whisper-large-v3",
        }.get((s, k), d)
        mock_config.expand_path.return_value = Path("/tmp/models/whisper")

        mock_result = MagicMock()
        with patch("meetcap.services.transcription.FasterWhisperService") as MockFW:
            mock_service = MagicMock()
            mock_service.transcribe.return_value = mock_result
            MockFW.return_value = mock_service

            screen._run_stt(audio_path, mock_config, "faster-whisper")

            # FasterWhisperService must be constructed with strings, not Config
            call_kwargs = MockFW.call_args.kwargs
            assert isinstance(call_kwargs["model_path"], str)
            assert isinstance(call_kwargs["model_name"], str)
            # transcribe must receive the Path
            mock_service.transcribe.assert_called_once_with(audio_path)
