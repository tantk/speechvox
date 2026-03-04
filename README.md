# SpeechVox

Local, GPU-accelerated push-to-talk speech-to-text for Windows. Runs entirely on your machine — no cloud, no API keys, no latency.

Press a hotkey, speak, release — your words are typed into any application.

## How it works

SpeechVox uses [Voxtral 4B Realtime](https://huggingface.co/mistralai/Voxtral-4B-Realtime) running on your NVIDIA GPU via a custom C/CUDA inference engine ([voxtral.c](https://github.com/antirez/voxtral.c)). The model is statically linked into a single executable.

- **Push-to-Talk**: Hold hotkey to record, release to transcribe and type
- **Continuous Streaming**: Toggle hotkey to start/stop live transcription
- **~7 GB VRAM** with Q4_K quantization, encoder warmup at load time

## Requirements

- Windows 10/11
- NVIDIA GPU with 6+ GB VRAM (8 GB recommended)
- CUDA 12+ installed
- Visual Studio 2022 Build Tools (MSVC)

## Quick start

```
git clone --recursive https://github.com/tantk/speechvox.git
cd speechvox
cargo build --release
./target/release/speechvox.exe
```

On first launch, the model manager opens automatically to download a quantized model (~3-5 GB depending on variant).

## Models

Downloaded via the built-in model manager. All models are Voxtral 4B Realtime in [VQF format](https://github.com/antirez/voxtral.c):

| Model | Quantization | Size | VRAM | Notes |
|-------|-------------|------|------|-------|
| **voxtral-4b-q4k** | Q4_K | ~3.0 GB | ~4 GB | Recommended. Best quality/size ratio |
| voxtral-4b-q6k | Q6_K | ~3.8 GB | ~5 GB | Near-lossless |
| voxtral-4b-q8 | Q8_0 | ~5.0 GB | ~7 GB | Reference quality |
| voxtral-4b-q4 | Q4_0 | ~3.1 GB | ~4 GB | Fastest dequant, slightly lower quality |

Models are hosted on [Hugging Face](https://huggingface.co/tantk/Voxtral-4B-Realtime-VQF).

## Configuration

Settings are stored in `speechvox.json` next to the executable.

- **Hotkey**: Right-click the overlay or tray icon > "Hotkey..." to set any key combo (e.g. F9, Ctrl+Space, Alt+Backquote)
- **Mode**: Switch between Push-to-Talk and Continuous Streaming via the tray/overlay menu
- **Overlay**: Drag the status indicator to reposition it. Position is saved on exit.

## Building from source

### Prerequisites

1. Rust toolchain (`rustup`)
2. CUDA Toolkit 12+ with `nvcc` on PATH
3. Visual Studio 2022 Build Tools with C++ workload

### Build

```
cargo build --release
```

The build system (`build.rs`) compiles the voxtral.c C/CUDA sources via the `cc` crate and statically links them. See [`docs/backend-build.md`](docs/backend-build.md) for detailed build pipeline documentation.

## Architecture

```
src/
  main.rs           - tao event loop, hotkey dispatch, tray/overlay menus
  engine.rs         - background thread running voxtral.c FFI inference
  audio.rs          - cpal mic capture, resample to 16kHz mono
  overlay.rs        - softbuffer always-on-top status overlay
  tray.rs           - system tray icon and menu
  hotkeys.rs        - global-hotkey crate wrapper, press/release detection
  hotkey_settings.rs - eframe/egui hotkey capture window
  model_manager.rs  - eframe/egui model download UI
  ffi.rs            - raw C FFI declarations for voxtral.c
  config.rs         - JSON config load/save
  models.rs         - model registry and file management
```

## License

MIT
