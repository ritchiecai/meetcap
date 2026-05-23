# meetcap

Offline meeting recorder & summarizer for macOS

## Features

- Records both system audio and microphone simultaneously
- 100% offline operation - no network connections
- Local transcription using Parakeet TDT (default, Apple Silicon) or Whisper
- Local summarization using Qwen3.5-4B via MLX (Apple Silicon native)
- Speaker diarization via sherpa-onnx (identifies who said what)
- LLM-based transcript refinement with custom hotwords (fix homophones, brand names, jargon)
- Simple CLI workflow: start recording → stop with hotkey → get transcript & summary

## Installation

### Prerequisites

1. **Install ffmpeg**:
   ```bash
   brew install ffmpeg
   ```

2. **Install BlackHole** (for system audio capture):
   - Download from: https://github.com/ExistentialAudio/BlackHole
   - Install the 2ch version

   > **Why BlackHole?** macOS doesn't provide a native way to capture system audio (sounds from other apps, video calls, etc.) as an input source. BlackHole creates a virtual audio device that routes system output into an input that meetcap can record.

3. **Configure Audio Routing**:

   Open **Audio MIDI Setup** (Applications → Utilities) and create two devices:

   **Step 1: Multi-Output Device** (so you can hear AND record system audio):
   - Click "+" → Create Multi-Output Device
   - Check: **Built-in Output** (or your preferred speakers/headphones)
   - Check: **BlackHole 2ch**
   - Right-click this device → "Use This Device For Sound Output"
   - Name it "Multi-Output Device" (or similar)

   **Step 2: Aggregate Device** (combines system audio + microphone):
   - Click "+" → Create Aggregate Device
   - Check: **BlackHole 2ch** (captures system audio routed from step 1)
   - Check: **Your Microphone** (Built-in Microphone, USB mic, etc.)
   - Set your microphone as **clock source** (right-click → Use This Device as Clock Source)
   - Check **Drift Correction** for your microphone
   - Name it "Aggregate Device" (or similar)

   > **How it works:** System audio flows to Multi-Output → BlackHole → Aggregate Device, where it combines with your microphone for recording.

4. **Models** (selected and downloaded during setup):
   - **Whisper models**: large-v3 (default), large-v3-turbo, or small
   - **Vosk models** (with speaker identification): small (507MB), standard (1.8GB), or gigaspeech (3.3GB)
   - **LLM models**: Qwen3.5-4B (default, ~2.9GB), Qwen3.5-9B (~5.6GB)

### Install meetcap

**macOS (Apple Silicon recommended):**
```bash
pip install "meetcap[mlx-stt]"
```

**Other platforms or Intel Macs:**
```bash
pip install "meetcap[stt]"
```

**With Vosk for speaker identification:**
```bash
pip install "meetcap[vosk-stt]"
```

**First-time setup:**
```bash
# Run interactive setup wizard (downloads models, tests permissions)
meetcap setup

# Quick system verification
meetcap verify

# See all available commands
meetcap --help
```

> **Note:** The `mlx-stt` extra includes MLX Whisper for faster transcription on Apple Silicon. The `stt` extra uses faster-whisper which works on all platforms.

## Usage

### Basic Recording

```bash
# Start recording (uses default/best audio device)
meetcap record

# Press ⌘+⇧+S or Ctrl-C to stop recording
```

### Process Existing Audio Files

```bash
# Transcribe and summarize an existing audio file
meetcap summarize path/to/meeting.m4a

# Specify output directory
meetcap summarize recording.m4a --out ./results
```

### Reprocess Recordings

```bash
# Reprocess both transcript and summary with different models
meetcap reprocess 2025_Jan_15_TeamStandup

# Reprocess only the summary (keep existing transcript)
meetcap reprocess 2025_Jan_15_TeamStandup --mode summary

# Use a different STT engine for transcription
meetcap reprocess /path/to/recording --stt mlx

# Use a custom LLM model for summarization
meetcap reprocess recording_dir --llm ~/.meetcap/models/custom_model.gguf

# Skip confirmation prompt
meetcap reprocess recording_dir --yes
```

The reprocess command allows you to regenerate transcripts and summaries with different models, useful for:
- Testing different STT engines (faster-whisper vs mlx-whisper)
- Trying different LLM models for better summaries
- Fixing issues with previous processing
- Experimenting with model settings

### Commands

```bash
# First-time setup (interactive wizard)
meetcap setup

# List available audio devices
meetcap devices

# Quick system verification
meetcap verify

# Start recording a meeting
meetcap record --device "Aggregate Device" --out ~/MyRecordings

# Process existing audio file (m4a, wav, mp3, etc.)
meetcap summarize samples/meeting.m4a --out ./processed

# Reprocess a recording with different models
meetcap reprocess 2025_Jan_15_TeamStandup --mode stt

# Refine an existing transcript with LLM + hotwords
meetcap refine 2025_Jan_15_TeamStandup --hotwords-file ~/.meetcap/hotwords.txt
```

### Configuration

Edit `~/.meetcap/config.toml` to customize:
- Default audio device
- STT engine selection (faster-whisper, mlx-whisper, vosk)
- Model settings (defaults to auto-downloaded models)
- Hotkey combinations
- Output directories

Models are automatically downloaded to `~/.meetcap/models/` on first use.

#### STT Engine Options

meetcap supports multiple speech-to-text engines:

1. **Faster-Whisper** (default on Intel Macs):
   - Best accuracy for general transcription
   - Models: large-v3, large-v3-turbo, small
   - Use: `--stt fwhisper`

2. **MLX-Whisper** (default on Apple Silicon):
   - Optimized for Apple Silicon performance
   - Models: large-v3-turbo, large-v3-mlx, small-mlx
   - Use: `--stt mlx`

3. **Vosk** (speaker identification):
   - Offline speech recognition with speaker diarization
   - Identifies different speakers in the meeting
   - Models: small (507MB), standard (1.8GB), gigaspeech (3.3GB)
   - Use: `--stt vosk`
   - Enable diarization in config:
   ```toml
   [models]
   stt_engine = "vosk"
   enable_speaker_diarization = true
   ```

Speaker identification improves summaries by attributing statements to specific speakers.

### Transcript Refinement (LLM Post-Editing)

ASR engines often mis-transcribe domain-specific terms (product names, code identifiers, foreign acronyms, people's names). meetcap can run a local LLM after STT to correct these errors **without re-running speech recognition**, while preserving timestamps and speaker labels.

**Quick start:**

```bash
# 1. Create a hotwords file with terms the ASR struggles with
cat > ~/.meetcap/hotwords.txt <<'EOF'
# One term per line. Lines starting with # are comments.
TCADP
Cognito
Jucoin
Kubernetes
PT Cyrameta
EOF

# 2. Enable refinement in ~/.meetcap/config.toml
#    (see [refinement] section below)

# 3. Refine an existing recording in place
meetcap refine ~/Recordings/2026_May_23_TeamSync --hotwords-file ~/.meetcap/hotwords.txt
```

When `[refinement].enabled = true`, refinement runs automatically as part of the normal pipeline (record / summarize / reprocess) — right after STT and before summarization, so the summary benefits from the corrected transcript.

**Configuration (`~/.meetcap/config.toml`):**

```toml
[refinement]
enabled = false                  # set to true to enable in the pipeline
mode = "diff"                    # "diff" (recommended) preserves segments & timestamps
backend = ""                     # "mlx-lm" or "omlx"; empty inherits from [llm].backend
model_name = ""                  # empty inherits from [llm].model_path / [omlx].model
temperature = 0.0
max_tokens = 1024
preserve_filler_words = true     # don't drop um/uh — only fix wrong words
hotwords_file = "~/.meetcap/hotwords.txt"
hotwords = []                    # inline list, merged with hotwords_file
segs_per_chunk = 12              # how many segments to send per LLM call
max_chars_per_chunk = 3000
keep_original = true             # save .transcript.original.{txt,json} backups
```

**What it does:**

- Splits the transcript into chunks of ≤ `segs_per_chunk` segments (or ≤ `max_chars_per_chunk` characters), keeping speaker turns intact.
- Sends each chunk plus the hotword list to the LLM with a strict JSON-only system prompt asking for `[{seg_index, original, corrected, reason}]`.
- Validates the response, applies word-level substring replacement on each affected segment, and rewrites `recording.transcript.{txt,json}` in place.
- Saves an audit log to `recording.corrections.json` listing every change with its reason.
- Backs up the pre-refinement files to `recording.transcript.original.{txt,json}` (when `keep_original = true`).

**Trade-offs vs. ASR-native hotword injection:**

- ✅ Works with any STT engine (mlx-whisper, faster-whisper, parakeet, vosk).
- ✅ No retranscription cost — runs on text only.
- ✅ Easy to iterate: edit hotwords, run `meetcap refine` again.
- ⚠️ The LLM can hallucinate; `mode = "diff"` and a low `temperature` (0.0) minimize this. Always review `corrections.json`.

See `docs/refinement.md` for a deeper dive, prompt design, and chunking algorithm.

## Permissions

Grant these permissions to your terminal app:
1. **Microphone**: System Preferences → Privacy & Security → Microphone
2. **Input Monitoring**: System Preferences → Privacy & Security → Input Monitoring

## Troubleshooting

### Audio Setup Issues

**No audio devices found:**
- Ensure BlackHole is installed and Audio MIDI Setup devices are created
- Run `meetcap devices` to list available devices
- Try restarting your terminal or system after setup

**Can't hear system audio during recording:**
- Check that Multi-Output Device is set as system output (right-click in Audio MIDI Setup)
- Ensure Built-in Output is checked in your Multi-Output Device
- Test by playing music - you should hear it through your speakers

**Recording only captures microphone or only system audio:**
- Verify Aggregate Device includes both BlackHole 2ch AND your microphone
- Check that microphone is set as clock source in Aggregate Device
- Ensure Drift Correction is enabled for your microphone
- Use `meetcap record --device "Aggregate Device"` to specify the device explicitly

**Volume control keys causing errors:**
- This is a known compatibility issue with some pynput versions - the app should continue working normally despite the error message

## Output Files

Each recording session creates a dedicated folder with all meeting artifacts organized together:

```
~/Recordings/meetcap/
├── 2025_Jan_15_TeamStandup/
│   ├── recording.opus                # Audio recording (OPUS format, default)
│   ├── recording.transcript.txt      # Plain text transcript
│   ├── recording.transcript.json     # Transcript with timestamps
│   └── recording.summary.md          # Meeting summary with AI-generated insights
├── 2025_Jan_16_ProductReview/
│   └── ... (same structure)
└── ...
```

### Folder Naming Convention

Meeting folders are automatically named using:
- **Date**: `YYYY_MMM_DD` format (e.g., `2025_Jan_15`)
- **Title**: AI-generated from meeting content in PascalCase (e.g., `TeamStandup`)

The AI analyzes your meeting transcript to generate a concise, descriptive title that captures the main topic discussed.

### Configuring Output Directory

The default output directory is `~/Recordings/meetcap/`. You can change this:

1. **During setup**: Run `meetcap setup` and specify your preferred directory
2. **In config file**: Edit `~/.meetcap/config.toml`:
   ```toml
   [paths]
   out_dir = "~/Documents/MyMeetings"  # Your custom path
   ```
3. **Via environment variable**:
   ```bash
   export MEETCAP_OUT_DIR="~/Documents/MyMeetings"
   ```

## License

MIT
