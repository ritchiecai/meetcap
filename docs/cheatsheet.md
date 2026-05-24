# meetcap 常用命令速查 (Cheatsheet)

> 适用版本：v3.0.0
> 仓库路径：`/Users/ritchie/Workspace/github/meetcap`
> 配置文件：`~/.meetcap/config.toml`
> 默认录制目录：`~/Recordings/meetcap`
> 默认模型缓存：`~/.meetcap/models`

---

## 1、开发测试场景

### 1.1 同步依赖

```bash
cd /Users/ritchie/Workspace/github/meetcap

# 仅核心 + dev 工具（pytest/ruff/mypy 等）
uv sync --extra dev

# 全可选 extras（开发推荐）
uv sync --extra dev --extra parakeet-stt --extra mlx-stt --extra stt --extra diarization --extra vosk-stt
```

| extra | 启用功能 | 关键依赖 |
|-------|---------|---------|
| `dev` | 测试 / lint / 类型检查 | pytest、pytest-cov、pytest-mock、ruff、mypy |
| `stt` | faster-whisper STT 后端 | faster-whisper |
| `mlx-stt` | mlx-whisper STT 后端（Apple Silicon） | mlx-whisper |
| `parakeet-stt` | Parakeet TDT 后端（默认 STT） | parakeet-mlx |
| `vosk-stt` | Vosk 离线 STT + 内置 diarization | vosk |
| `diarization` | sherpa-onnx 说话人分离 | sherpa-onnx、librosa、soundfile |

### 1.2 单元测试

```bash
# 全量测试 + 覆盖率（pyproject 已配 --cov-fail-under=73）
uv run pytest

# 单文件
uv run pytest tests/test_recorder.py -v

# 单函数
uv run pytest tests/test_recorder.py::test_start_recording_wav -v

# 关键字过滤
uv run pytest -k "diarization and not slow"

# 只跑上次失败的
uv run pytest --lf

# 关闭覆盖率门槛（调试时）
uv run pytest --no-cov

# 显示 print/log
uv run pytest -s

# 指定覆盖率报告格式
uv run pytest --cov=meetcap --cov-report=html   # 输出到 htmlcov/
```

### 1.3 Lint / 格式 / 类型检查

```bash
uv run ruff check meetcap tests             # 静态检查
uv run ruff check meetcap tests --fix       # 自动修复
uv run ruff format meetcap tests            # 自动格式化
uv run ruff format --check meetcap tests    # 仅校验
uv run mypy meetcap                         # 类型检查
```

### 1.4 TUI 调试

```bash
# 终端 A：开发者控制台（看日志/事件）
uv run textual console

# 终端 B：dev 模式启动 TUI，自动连到控制台
uv run textual run --dev "meetcap.tui.app:MeetcapApp"
```

### 1.5 构建与本地包验证

```bash
# 构建 wheel + sdist（产物在 dist/）
uv build

# 用本地 wheel 跑 verify（隔离环境）
uv run --isolated --with dist/meetcap-3.0.0-py3-none-any.whl meetcap verify
```

---

## 2、本地安装场景

### 2.1 推荐：uv tool 全局隔离（日常使用）

```bash
cd /Users/ritchie/Workspace/github/meetcap

# 仅核心
uv tool install . --force

# 带常用 extras（推荐，开会能直接用）
uv tool install . --force \
  --with "meetcap[stt,mlx-stt,parakeet-stt,diarization]"

# 反复刷新源码改动
uv tool install . --force

# 卸载
uv tool uninstall meetcap
```

> 命令位置：`~/.local/bin/meetcap`，确保在 `$PATH` 里。

### 2.2 开发者可编辑安装（边改边用）

```bash
cd /Users/ritchie/Workspace/github/meetcap
uv sync --extra dev --extra parakeet-stt --extra mlx-stt --extra diarization

# 调用方式
uv run meetcap verify
# 或
source .venv/bin/activate && meetcap verify
```

### 2.3 wheel + pipx（分发给同事）

```bash
cd /Users/ritchie/Workspace/github/meetcap
uv build

pipx install dist/meetcap-3.0.0-py3-none-any.whl --force
# 带 extras
pipx install "dist/meetcap-3.0.0-py3-none-any.whl[stt,mlx-stt,parakeet-stt,diarization]" --force

pipx uninstall meetcap
```

### 2.4 安装后验收三连

```bash
which meetcap
meetcap --version
meetcap verify
```

---

## 3、首次配置场景

```bash
# 交互式 7 步配置向导（音频设备 / STT / LLM / 路径等）
uv run meetcap setup

# 列出系统音频设备
uv run meetcap devices

# 体检：ffmpeg / 模型 / 权限
uv run meetcap verify
```

配置文件位置：`~/.meetcap/config.toml`，模型缓存：`~/.meetcap/models/`。

---

## 4、日常使用场景

### 4.1 启动 TUI

```bash
uv run meetcap                           # 默认进入 home 屏幕
uv run meetcap --screen record           # 直接进入录制屏
uv run meetcap --screen history          # 直接进入历史屏
```

TUI 快捷键：`r` 录制 / `h` 历史 / `s` 设置 / `?` 帮助 / `q` 退出。

### 4.2 录制会议

```bash
# 默认录制（按 ⌘+⇧+S 或 Ctrl-C 停止）
uv run meetcap record

# 强制覆盖某些参数（不改配置文件）
MEETCAP_STT_ENGINE=mlx-whisper \
MEETCAP_DEVICE="Aggregate Device" \
  uv run meetcap record
```

### 4.3 分析已有音频文件（第三方录音）

```bash
# 默认：在音频所在目录写产物（*.transcript.txt / *.transcript.json / *.summary.md）
uv run meetcap summarize ~/Downloads/meeting.m4a

# 指定输出目录
uv run meetcap summarize ~/Downloads/meeting.m4a -o ~/Recordings/meetcap/imports

# 指定 STT 引擎（中文场景推荐 mlx）
uv run meetcap summarize ~/Downloads/meeting.m4a --stt mlx --no-tui

# 临时换 LLM 模型
uv run meetcap summarize ~/Downloads/meeting.m4a \
  --llm "mlx-community/Qwen3.5-4B-OptiQ-4bit"
```

支持格式：m4a / wav / mp3 / opus / flac 等所有 ffmpeg 可解码格式。

### 4.4 重跑历史录制（meetcap 录制目录）

```bash
# 默认全流程（STT + 摘要）
uv run meetcap reprocess ~/Recordings/meetcap/2026-05-23_SomeMeeting

# 只重跑摘要（跳过 STT）
uv run meetcap reprocess <dir> --mode summary

# 换 STT 引擎重跑
uv run meetcap reprocess <dir> --stt mlx --yes --no-tui

# 换 LLM
uv run meetcap reprocess <dir> --llm "mlx-community/Qwen3.5-9B-OptiQ-4bit"

# 跳过确认 + 关闭 TUI（脚本化）
uv run meetcap reprocess <dir> --yes --no-tui
```

> `--mode` 可选 `stt`（默认，全流程）或 `summary`（仅重跑摘要）。

---

## 5、关键环境变量速查

> 仅以下变量可通过环境变量覆盖；`[llm]` 节当前**不支持**环境变量覆盖（只能改 toml）。

| 变量 | 对应配置 | 示例 |
|------|---------|------|
| `MEETCAP_DEVICE` | `[audio].preferred_device` | `"Aggregate Device"` |
| `MEETCAP_AUDIO_FORMAT` | `[audio].format` | `opus` / `wav` / `flac` |
| `MEETCAP_OPUS_BITRATE` | `[audio].opus_bitrate` | `32` |
| `MEETCAP_STT_ENGINE` | `[models].stt_engine` | `parakeet` / `mlx-whisper` / `faster-whisper` / `vosk` / `whispercpp` |
| `MEETCAP_LLM_MODEL` | `[models].llm_model_name` | `mlx-community/Qwen3.5-4B-OptiQ-4bit` |
| `MEETCAP_OUT_DIR` | `[paths].out_dir` | `~/Recordings/meetcap` |
| `MEETCAP_HOTKEY` | `[hotkey].stop` | `<cmd>+<shift>+s` |
| `MEETCAP_HOTKEY_PREFIX` | `[hotkey].prefix` | `<ctrl>+a` |
| `MEETCAP_ENABLE_DIARIZATION` | `[models].enable_speaker_diarization` | `true` / `false` |
| `MEETCAP_DIARIZATION_BACKEND` | `[models].diarization_backend` | `sherpa` / `vosk` |
| `MEETCAP_MEMORY_*` | `[memory].*` | 详见 `Config._apply_env_overrides` |

---

## 6、oMLX 后端配置速查

`~/.meetcap/config.toml`：

```toml
[models]
llm_model_name = "mlx-community/Qwen3.5-4B-OptiQ-4bit"   # 必须是 oMLX 已加载的模型 ID

[llm]
backend = "omlx"
omlx_base_url = "http://localhost:8000/v1"
omlx_api_key = "ipc@IND123"
omlx_timeout = 600
temperature = 0.4
max_tokens = 4096
enable_thinking = false
thinking_budget = 512
```

### 6.1 oMLX 服务端自测

```bash
# 列出已加载模型
curl -H "Authorization: Bearer ipc@IND123" http://localhost:8000/v1/models

# 一次性 chat completion ping
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ipc@IND123" \
  -d '{
    "model": "mlx-community/Qwen3.5-4B-OptiQ-4bit",
    "messages": [{"role":"user","content":"ping"}],
    "max_tokens": 16
  }'
```

---

## 7、音频问题排障速查

### 7.1 音频文件健康检查

```bash
# 时长 / 通道 / 比特率
ffprobe -v error -show_format -show_streams <audio_file>

# 音量检测（mean_volume / max_volume；< -50dB 基本静音）
ffmpeg -i <audio_file> -af "volumedetect" -f null /dev/null 2>&1 \
  | grep -E "mean_volume|max_volume|n_samples"

# 试听
afplay <audio_file>

# 转 16kHz mono wav 用于其他工具
ffmpeg -i <audio_file> -ar 16000 -ac 1 /tmp/audio_test.wav -y
```

### 7.2 STT 交叉验证（怀疑某引擎异常时）

```bash
# 用 mlx-whisper 直接跑（绕过 meetcap，纯 Python）
uv run python -c "
import mlx_whisper
result = mlx_whisper.transcribe(
    '/tmp/audio_test.wav',
    path_or_hf_repo='mlx-community/whisper-large-v3-turbo',
    word_timestamps=False,
    verbose=False,
)
print('lang:', result.get('language'))
print('segments:', len(result.get('segments', [])))
print(result.get('text', ''))
"
```

### 7.3 常见错误对应

| 现象 | 处置 |
|------|------|
| `parakeet-mlx not installed` | `uv sync --extra parakeet-stt` 或切其他 STT 引擎 |
| transcript 文件 0 字节 | 看 `*.transcript.json` 的 `language`；中文录音不要用 Parakeet，改 mlx-whisper |
| ffmpeg 找不到设备 | `meetcap devices` 看 index；BlackHole/Aggregate Device 要在"音频 MIDI 设置"建好 |
| 麦克风权限失败 | 系统设置 → 隐私与安全 → 麦克风/输入监控 勾选终端 App |
| oMLX 调用 401 | 检查 `omlx_api_key` |
| oMLX connection refused | oMLX 服务没起或 `omlx_base_url` 错 |
| oMLX timeout | 调大 `omlx_timeout`（默认 300，长会议建议 600+） |
| 覆盖率不到 73% | `uv run pytest --cov=meetcap --cov-report=term-missing` 看缺哪几行 |

---

## 8、产物文件结构速查

`meetcap record` 完成后的目录（命名 `YYYY-MM-DD_<PascalTitle>` 或 `YYYYMMDD-HHMMSS-temp`）：

```
2026-05-23_CookingLesson/
├── recording.opus              # 主音频（默认 opus 32kbps；可配 wav/flac）
├── recording.transcript.txt    # 纯文本转写（一行一句）
├── recording.transcript.json   # 带时间戳/speaker_id 的结构化 JSON
├── recording.summary.md        # LLM 生成的 8 段结构化摘要
└── notes.md                    # 用户手动备注（可选，自动并入摘要 prompt）
```

`meetcap summarize <audio>` 产物（默认在音频所在目录）：

```
<audio_dir>/
├── <basename>.transcript.txt
├── <audio_basename>.transcript.json
└── <audio_basename>.summary.md
```

---

## 9、常用一键流程

### 9.1 中文会议全自动（已录好的音频）

```bash
uv run meetcap summarize ~/Downloads/meeting.m4a --stt mlx --no-tui
```

### 9.2 重跑某个目录的摘要（音频不变）

```bash
uv run meetcap reprocess ~/Recordings/meetcap/<dir> --mode summary --yes --no-tui
```

### 9.3 升级本地工具版本

```bash
cd /Users/ritchie/Workspace/github/meetcap
git pull
uv tool install . --force --with "meetcap[stt,mlx-stt,parakeet-stt,diarization]"
meetcap --version
meetcap verify
```

### 9.4 完整开发循环

```bash
cd /Users/ritchie/Workspace/github/meetcap
uv sync --extra dev --extra parakeet-stt --extra mlx-stt --extra diarization
uv run ruff check meetcap tests --fix
uv run ruff format meetcap tests
uv run mypy meetcap
uv run pytest
```

---

## 10、附：四种安装方式对比

| 方式 | 命令隔离 | 源码改动即生效 | 适合场景 |
|------|---------|---------------|---------|
| `uv tool install` | ✅ 强隔离 | ❌（要 reinstall） | 当成日常工具 |
| `uv sync` (editable) | 项目级 venv | ✅ | 本地开发调试 |
| `uv build` + `pipx` | ✅ 强隔离 | ❌ | 打包分发 |
| `pip install -e .` | 取决于环境 | ✅ | 没装 uv/pipx 的兜底 |

---

_最后更新：2026-05-23_
