"""configuration management for meetcap"""

import os
import sys
from enum import Enum
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from rich.console import Console

from meetcap.utils.logger import logger

console = Console()


class AudioFormat(str, Enum):
    """Supported audio recording formats."""

    WAV = "wav"
    OPUS = "opus"
    FLAC = "flac"


class Config:
    """manages application configuration"""

    DEFAULT_CONFIG = {
        "audio": {
            "preferred_device": "Aggregate Device",
            "sample_rate": 48000,
            "channels": 2,
            "format": "opus",  # recording format: wav, opus, or flac (default opus for space efficiency)
            "opus_bitrate": 32,  # opus bitrate in kbps (6-510)
            "flac_compression_level": 5,  # flac compression level (0-8)
        },
        "recording": {
            "default_auto_stop": 0,  # Default scheduled stop time in minutes (0 = no auto stop)
        },
        "hotkey": {
            "stop": "<cmd>+<shift>+s",
            "prefix": "<ctrl>+a",  # prefix key for timer operations
        },
        "models": {
            "stt_engine": "parakeet",  # stt engine: parakeet, faster-whisper, mlx-whisper, vosk, or whispercpp
            "stt_model_name": "large-v3",  # whisper model name for auto-download
            "stt_model_path": "~/.meetcap/models/whisper-large-v3",  # will be created automatically
            "mlx_stt_model_name": "mlx-community/whisper-large-v3-turbo",  # mlx whisper model
            "mlx_stt_model_path": "~/.meetcap/models/mlx-whisper",  # mlx models directory
            "parakeet_model_name": "mlx-community/parakeet-tdt-0.6b-v3",  # parakeet model
            "vosk_model_name": "vosk-model-en-us-0.22",  # vosk model name
            "vosk_model_path": "~/.meetcap/models/vosk/vosk-model-en-us-0.22",  # vosk model directory
            "vosk_spk_model_path": "~/.meetcap/models/vosk/vosk-model-spk-0.4",  # speaker model directory
            "enable_speaker_diarization": True,  # enable speaker identification (now default)
            "diarization_backend": "sherpa",  # diarization backend: sherpa or vosk
            "sherpa_num_speakers": -1,  # expected speaker count (-1 for auto)
            "sherpa_cluster_threshold": 0.90,  # clustering threshold (higher = fewer speakers)
            "llm_model_name": "mlx-community/Qwen3.5-2B-OptiQ-4bit",  # mlx llm model repo id
        },
        "paths": {
            "out_dir": "~/Recordings/meetcap",
            "models_dir": "~/.meetcap/models",  # directory for auto-downloaded models
        },
        "llm": {
            "backend": "mlx-lm",  # llm backend: "mlx-lm" (in-process) or "omlx" (server)
            "omlx_base_url": "http://localhost:8000/v1",  # oMLX server API endpoint
            "omlx_api_key": "",  # oMLX API key (empty = no auth)
            "omlx_timeout": 300,  # oMLX request timeout in seconds
            "temperature": 0.4,
            "max_tokens": 4096,
            "enable_thinking": False,  # disable thinking by default for faster output
            "thinking_budget": 512,  # max thinking tokens when thinking is enabled
        },
        "telemetry": {
            "disable": True,
        },
        "memory": {
            "aggressive_gc": True,  # enable aggressive garbage collection between models
            "enable_monitoring": False,  # enable memory monitoring and reporting
            "memory_report": False,  # print detailed memory report after processing
            "warning_threshold": 80,  # memory pressure warning threshold (percentage)
            "critical_threshold": 90,  # memory pressure critical threshold (percentage)
            "auto_fallback": True,  # automatic model fallback when memory is constrained
            "explicit_lifecycle": True,  # force explicit model loading/unloading
        },
        "notes": {
            "enable": True,  # enable manual notes feature
            "template": "# Meeting Notes\n\n*Add your notes here during or after the meeting*\n\n*This file will be included in the final summary*\n",  # template for new notes files
            "filename": "notes.md",  # default notes file name
        },
    }

    def __init__(self, config_path: Path | None = None):
        """
        initialize config.

        args:
            config_path: path to config file (default: ~/.meetcap/config.toml)
        """
        if config_path is None:
            config_path = Path.home() / ".meetcap" / "config.toml"

        self.config_path = config_path
        # deep copy to ensure test isolation
        import copy

        self.config = copy.deepcopy(self.DEFAULT_CONFIG)

        # load from file if exists
        if self.config_path.exists():
            self._load_from_file()
            self._migrate_config()  # apply any necessary migrations

        # apply environment variable overrides
        self._apply_env_overrides()

    def _load_from_file(self) -> None:
        """load configuration from toml file."""
        try:
            with open(self.config_path, "rb") as f:
                file_config = tomllib.load(f)

            # merge with defaults
            self._deep_merge(self.config, file_config)

        except Exception as e:
            console.print(f"[yellow]warning: failed to load config: {e}[/yellow]")

    def _apply_env_overrides(self) -> None:
        """apply environment variable overrides."""
        env_mapping = {
            "MEETCAP_DEVICE": ("audio", "preferred_device"),
            "MEETCAP_SAMPLE_RATE": ("audio", "sample_rate", int),
            "MEETCAP_CHANNELS": ("audio", "channels", int),
            "MEETCAP_AUDIO_FORMAT": ("audio", "format"),
            "MEETCAP_OPUS_BITRATE": ("audio", "opus_bitrate", int),
            "MEETCAP_FLAC_COMPRESSION": ("audio", "flac_compression_level", int),
            "MEETCAP_HOTKEY": ("hotkey", "stop"),
            "MEETCAP_HOTKEY_PREFIX": ("hotkey", "prefix"),
            "MEETCAP_STT_ENGINE": ("models", "stt_engine"),
            "MEETCAP_STT_MODEL": ("models", "stt_model_path"),
            "MEETCAP_VOSK_MODEL": ("models", "vosk_model_name"),
            "MEETCAP_VOSK_MODEL_PATH": ("models", "vosk_model_path"),
            "MEETCAP_VOSK_SPK_MODEL": ("models", "vosk_spk_model_path"),
            "MEETCAP_ENABLE_DIARIZATION": ("models", "enable_speaker_diarization", bool),
            "MEETCAP_MLX_STT_MODEL": ("models", "mlx_stt_model_name"),
            "MEETCAP_PARAKEET_MODEL": ("models", "parakeet_model_name"),
            "MEETCAP_DIARIZATION_BACKEND": ("models", "diarization_backend"),
            "MEETCAP_SHERPA_NUM_SPEAKERS": ("models", "sherpa_num_speakers", int),
            "MEETCAP_SHERPA_THRESHOLD": ("models", "sherpa_cluster_threshold", float),
            "MEETCAP_LLM_MODEL": ("models", "llm_model_name"),
            "MEETCAP_OUT_DIR": ("paths", "out_dir"),
            # memory management settings
            "MEETCAP_MEMORY_AGGRESSIVE_GC": (
                "memory",
                "aggressive_gc",
                lambda x: x.lower() == "true",
            ),
            "MEETCAP_MEMORY_MONITORING": (
                "memory",
                "enable_monitoring",
                lambda x: x.lower() == "true",
            ),
            "MEETCAP_MEMORY_REPORT": ("memory", "memory_report", lambda x: x.lower() == "true"),
            "MEETCAP_MEMORY_WARNING_THRESHOLD": ("memory", "warning_threshold", int),
            "MEETCAP_MEMORY_AUTO_FALLBACK": (
                "memory",
                "auto_fallback",
                lambda x: x.lower() == "true",
            ),
            # manual notes settings
            "MEETCAP_NOTES_ENABLE": (
                "notes",
                "enable",
                lambda x: x.lower() == "true",
            ),
            "MEETCAP_NOTES_TEMPLATE": ("notes", "template"),
            "MEETCAP_NOTES_FILENAME": ("notes", "filename"),
        }

        for env_var, path_spec in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                # parse type if specified
                if len(path_spec) == 3:
                    section, key, type_func = path_spec
                    try:
                        value = type_func(value)
                    except ValueError:
                        logger.warning(f"ignoring invalid env var {env_var}={value}")
                        continue
                else:
                    section, key = path_spec

                # set value
                if section in self.config:
                    self.config[section][key] = value

    def _deep_merge(self, base: dict, update: dict) -> None:
        """recursively merge update dict into base dict."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _migrate_config(self) -> None:
        """apply config migrations for backward compatibility."""
        # migrate GGUF-based config to MLX-based config
        models = self.config.get("models", {})
        needs_migration = False

        # check for old llm_gguf_path key
        if "llm_gguf_path" in models:
            console.print("[dim]migrating from GGUF to MLX llm model...[/dim]")
            del self.config["models"]["llm_gguf_path"]
            self.config.setdefault("models", {}).setdefault(
                "llm_model_name", "mlx-community/Qwen3.5-2B-OptiQ-4bit"
            )
            needs_migration = True

        # check for llm_model_name ending in .gguf
        if models.get("llm_model_name", "").endswith(".gguf"):
            console.print("[dim]migrating .gguf model name to MLX model...[/dim]")
            self.config["models"]["llm_model_name"] = "mlx-community/Qwen3.5-2B-OptiQ-4bit"
            needs_migration = True

        if needs_migration:
            self.save()

        # migrate audio format to opus for existing configs without format key
        # this includes both: missing format key AND format="wav"
        current_format = self.config.get("audio", {}).get("format")
        if current_format is None or current_format == "wav":
            if current_format is None:
                console.print(
                    "[dim]setting audio format to opus (98% space savings vs previous default)...[/dim]"
                )
            else:
                console.print(
                    "[dim]migrating audio format from wav to opus (98% space savings)...[/dim]"
                )

            self.config.setdefault("audio", {}).setdefault("format", "opus")

            # also set the opus_bitrate and flac_compression_level defaults if not present
            self.config.setdefault("audio", {}).setdefault("opus_bitrate", 32)
            self.config.setdefault("audio", {}).setdefault("flac_compression_level", 5)

            # save the migration immediately
            self.save()

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        get configuration value.

        args:
            section: config section name
            key: config key name
            default: default value if not found

        returns:
            configuration value or default
        """
        return self.config.get(section, {}).get(key, default)

    def get_section(self, section: str) -> dict[str, Any]:
        """
        get entire configuration section.

        args:
            section: section name

        returns:
            section dict or empty dict
        """
        return self.config.get(section, {})

    def save(self) -> None:
        """save current configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # convert to toml format
        import toml

        with open(self.config_path, "w") as f:
            toml.dump(self.config, f)

        console.print(f"[green]✓[/green] config saved to {self.config_path}")

    def create_default_config(self) -> None:
        """create default config file if it doesn't exist."""
        if not self.config_path.exists():
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self.save()
            console.print(f"[green]✓[/green] created default config: {self.config_path}")
            console.print("[yellow]edit this file to customize your settings[/yellow]")

    def is_configured(self) -> bool:
        """check if meetcap has been configured (setup wizard completed)."""
        return self.config_path.exists()

    def expand_path(self, path_str: str) -> Path:
        """
        expand path with ~ and environment variables.

        args:
            path_str: path string possibly with ~ or env vars

        returns:
            expanded path object
        """
        return Path(os.path.expanduser(os.path.expandvars(path_str)))
