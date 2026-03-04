# SpeechVox Backend Build System

This document describes how the SpeechVox Rust application compiles and links the Voxtral 4B C inference engine, including CUDA support, FFI declarations, and the full build pipeline.

---

## 1. Overview

SpeechVox is a Rust application that statically links a C inference engine (`voxtral.c`) for speech-to-text. The build pipeline:

```
voxtral.c (11 C source files)
    ↓  cc crate (build.rs)
libvoxtral.a / voxtral.lib (static library)
    ↓  rustc linker
speechvox.exe (final binary, statically linked)
    + CUDA libs (cudart_static, cublas, cublasLt, cuda)
    + MSVC runtime (msvcprt)
    + Windows system libs (user32, shell32, advapi32, ole32)
```

---

## 2. Cargo.toml

### Dependencies

```toml
[package]
name = "speechvox"
version = "0.1.0"
edition = "2021"

[dependencies]
cpal = "0.15"                    # Audio capture (WASAPI/CoreAudio/ALSA)
tao = "0.30"                     # Cross-platform event loop
softbuffer = "0.4"               # Software framebuffer for overlay
enigo = "0.2"                    # Simulated keyboard input (typing text)
tray-icon = "0.17"               # System tray integration
image = { version = "0.25", default-features = false, features = ["png"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
parking_lot = "0.12"             # Fast mutexes
crossbeam-channel = "0.5"        # Multi-producer multi-consumer channels
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

[dependencies.windows]
version = "0.58"
features = [
    "Win32_System_Threading",
    "Win32_Foundation",
    "Win32_Security",
    "Win32_UI_WindowsAndMessaging",
    "Win32_UI_Input_KeyboardAndMouse",
]

[build-dependencies]
cc = "1.2"                       # C compiler integration
```

No custom `.cargo/config.toml` exists. All build configuration lives in `build.rs`.

---

## 3. Build Script (build.rs)

The build script uses the `cc` crate to compile 11 C source files into a static library and links CUDA, MSVC, and Windows system libraries.

### C Source Files

All sourced from `C:\dev\mistralhack\voxtral.c`:

| File | Purpose |
|------|---------|
| `voxtral.c` | Core inference: model loading, streaming API, token generation loop |
| `voxtral_kernels.c` | CPU compute kernels: matmul, RMSNorm, RoPE, SwiGLU, softmax |
| `voxtral_audio.c` | Mel spectrogram: FFT, Hann window, Slaney mel filterbank |
| `voxtral_encoder.c` | Audio encoder: conv stem, 32 transformer layers, adapter |
| `voxtral_decoder.c` | LLM decoder: 26 transformer layers, GQA, adaptive norm |
| `voxtral_tokenizer.c` | Tekken BPE tokenizer: 131K vocab, token decode |
| `voxtral_safetensors.c` | Safetensors format parser (unquantized model loading) |
| `voxtral_quant_loader.c` | VQF format loader: mmap, tensor lookup, BF16 conversion |
| `voxtral_quant_kernels.c` | CPU quantized GEMV: Q8_0, Q4_0, Q4_K dequant+matmul |
| `voxtral_cuda.c` | CUDA acceleration: attention, matmul, encoder/decoder pipelines |
| `voxtral_cuda_quant.c` | CUDA quantized kernels: GPU GEMV, weight upload/cache |

### Compiler Configuration

```rust
let mut build = cc::Build::new();

// Disable compiler warnings (third-party C code)
build.warnings(false);

// Optimization
build.flag_if_supported("/O2");   // MSVC
build.flag_if_supported("-O3");   // GCC/Clang

// Preprocessor defines
build.define("USE_CUDA", None);
build.define("VOX_CUDA_ARCH", "sm_89");  // RTX 4070 Ti

// Include paths
build.include("C:\\dev\\mistralhack\\voxtral.c");
build.include("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include");

// Compile all 11 source files
for src in &sources {
    build.file(format!("C:\\dev\\mistralhack\\voxtral.c\\{}", src));
}

// Output static library named "voxtral"
build.compile("voxtral");
```

### Library Linking

#### CUDA Libraries

```rust
// CUDA toolkit search path
println!("cargo:rustc-link-search=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\lib\\x64");

// Static CUDA runtime (no cudart DLL dependency)
println!("cargo:rustc-link-lib=cudart_static");

// cuBLAS for matrix multiplication
println!("cargo:rustc-link-lib=cublas");
println!("cargo:rustc-link-lib=cublasLt");

// CUDA driver API
println!("cargo:rustc-link-lib=cuda");
```

#### MSVC C++ Runtime

```rust
// Required for C++ STL symbols used by CUDA headers
println!("cargo:rustc-link-search=C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.44.35207\\lib\\x64");
println!("cargo:rustc-link-lib=msvcprt");
```

#### Windows System Libraries

```rust
println!("cargo:rustc-link-lib=user32");    // Window management
println!("cargo:rustc-link-lib=shell32");   // Shell operations
println!("cargo:rustc-link-lib=advapi32");  // Registry, security
println!("cargo:rustc-link-lib=ole32");     // COM
```

### Rebuild Triggers

```rust
println!("cargo:rerun-if-changed=build.rs");
for src in &sources {
    println!("cargo:rerun-if-changed=C:\\dev\\mistralhack\\voxtral.c\\{}", src);
}
```

Cargo reruns the build script if `build.rs` or any of the 11 C source files change.

---

## 4. FFI Boundary (ffi.rs)

The Rust-C boundary is defined in `src/ffi.rs`. All functions use C calling convention and opaque `*mut c_void` pointers for the model context and stream handles.

### Model Lifecycle

```rust
extern "C" {
    /// Load model from directory (consolidated.vqf or consolidated.safetensors + tekken.json).
    /// Returns opaque vox_ctx_t pointer, or null on failure.
    pub fn vox_load(model_dir: *const c_char) -> *mut c_void;

    /// Free model context and all associated memory.
    pub fn vox_free(ctx: *mut c_void);

    /// Set transcription delay in milliseconds (default 480ms = 6 tokens).
    pub fn vox_set_delay(ctx: *mut c_void, delay_ms: c_int);
}
```

### CUDA Support

```rust
extern "C" {
    /// Returns non-zero if CUDA is available.
    pub fn vox_cuda_available() -> c_int;

    /// Returns CUDA device name string (e.g. "NVIDIA GeForce RTX 4070 Ti").
    pub fn vox_cuda_device_name() -> *const c_char;

    /// Pre-populate GPU caches at load time (shifts first-token cost).
    pub fn vox_cuda_warmup_encoder(ctx: *mut c_void);
}
```

### Batch Transcription (Mode 1)

```rust
extern "C" {
    /// Transcribe audio buffer in one shot. Returns malloc'd C string (caller must free).
    pub fn vox_transcribe_audio(
        ctx: *mut c_void,
        samples: *const c_float,
        n_samples: c_int,
    ) -> *mut c_char;
}
```

### Streaming API (Modes 2 & 3)

```rust
extern "C" {
    /// Create a streaming context. Returns opaque vox_stream_t pointer.
    pub fn vox_stream_init(ctx: *mut c_void) -> *mut c_void;

    /// Feed PCM samples (16kHz mono f32) to the stream.
    pub fn vox_stream_feed(s: *mut c_void, samples: *const c_float, n: c_int) -> c_int;

    /// Flush pending tokens (may miss last word in CUDA chunked encoder).
    pub fn vox_stream_flush(s: *mut c_void) -> c_int;

    /// Finalize stream: flush + mel finish + final encoder/decoder pass.
    /// Permanently closes the stream (sets finished=1).
    pub fn vox_stream_finish(s: *mut c_void) -> c_int;

    /// End current utterance: like finish but keeps stream alive.
    /// Call vox_stream_reset_encoder() before feeding the next utterance.
    pub fn vox_stream_end_utterance(s: *mut c_void) -> c_int;

    /// Retrieve decoded text tokens. Returns number of tokens written to out[].
    pub fn vox_stream_get(s: *mut c_void, out: *mut *const c_char, max: c_int) -> c_int;

    /// Reset encoder state for next utterance (split cache trick).
    /// Preserves decoder KV cache for semantic continuity.
    pub fn vox_stream_reset_encoder(s: *mut c_void) -> c_int;

    /// Free stream context.
    pub fn vox_stream_free(s: *mut c_void);

    /// Set minimum processing interval in seconds (lower = more responsive).
    pub fn vox_set_processing_interval(s: *mut c_void, seconds: c_float);
}
```

### Verbosity & Memory

```rust
extern "C" {
    /// Global verbosity level (0=quiet, 1=info, 2=debug).
    pub static mut vox_verbose: c_int;

    /// C stdlib free (to release strings from vox_transcribe_audio).
    pub fn free(ptr: *mut c_void);
}
```

---

## 5. Engine Thread FFI Usage (engine.rs)

The engine thread also declares CRT functions for environment setup:

```rust
extern "C" {
    fn freopen(filename: *const c_char, mode: *const c_char, stream: *mut c_void) -> *mut c_void;
    fn __acrt_iob_func(index: u32) -> *mut c_void;
    fn _putenv_s(name: *const c_char, value: *const c_char) -> c_int;
}
```

### Why `_putenv_s` Instead of `std::env::set_var`

The C code uses `getenv()` from the CRT, which reads from the CRT's own environment block. Rust's `std::env::set_var` calls Win32 `SetEnvironmentVariable`, which modifies the Win32 process environment block. These are **separate** on Windows. Using `_putenv_s` ensures the C code's `getenv("VOX_CUDA_FAST")` sees the value.

### Initialization Sequence

```rust
// 1. Set CUDA environment variable via CRT
unsafe { _putenv_s(c"VOX_CUDA_FAST".as_ptr(), c"1".as_ptr()); }

// 2. Redirect C stderr to log file
unsafe {
    let stderr = __acrt_iob_func(2);
    freopen(c"voxtral_stderr.log".as_ptr(), c"w".as_ptr(), stderr);
}

// 3. Set verbosity
unsafe { ffi::vox_verbose = 2; }

// 4. Load model
let ctx = unsafe { ffi::vox_load(model_dir.as_ptr()) };
assert!(!ctx.is_null());

// 5. Check CUDA and warmup
if unsafe { ffi::vox_cuda_available() } != 0 {
    unsafe { ffi::vox_cuda_warmup_encoder(ctx); }
}
```

### Memory Management Pattern

- **Model context**: C allocates via `vox_load`, Rust frees via `vox_free`
- **Stream context**: C allocates via `vox_stream_init`, Rust frees via `vox_stream_free`
- **Transcription strings**: C allocates via `malloc` inside `vox_transcribe_audio`, Rust frees via C `free()`
- **Token strings from `vox_stream_get`**: Pointers into tokenizer's internal string table, **not** freed by caller

---

## 6. Build Environment

| Component | Version / Path |
|-----------|---------------|
| Platform | Windows 11 Pro (x64) |
| Rust | Edition 2021 |
| C Compiler | MSVC (Visual Studio 2022 Community) |
| MSVC Version | 14.44.35207 |
| CUDA Toolkit | v13.0 |
| CUDA Architecture | sm_89 (RTX 4070 Ti) |
| Voxtral Source | `C:\dev\mistralhack\voxtral.c` |
| CUDA Path | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0` |

---

## 7. Build Commands

```bash
# Normal build
cargo build --release

# Build with verbose cc output (shows compiler invocations)
cargo build --release -vv

# Run the main application
cargo run --release

# Run the multi-utterance test binary
cargo run --release --bin test_multi_utterance -- english.wav chinese.wav
```

The `cc` crate handles all C compilation automatically. No separate CMake or Makefile step is needed.
