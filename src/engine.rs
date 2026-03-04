use crate::ffi;
use crossbeam_channel::{Receiver, Sender};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use tracing::info;

extern "C" {
    fn freopen(filename: *const std::os::raw::c_char, mode: *const std::os::raw::c_char, stream: *mut std::os::raw::c_void) -> *mut std::os::raw::c_void;
    fn __acrt_iob_func(index: u32) -> *mut std::os::raw::c_void;
    fn _putenv_s(name: *const std::os::raw::c_char, value: *const std::os::raw::c_char) -> std::os::raw::c_int;
}

pub enum EngineCommand {
    Transcribe(Vec<f32>),
    FeedAudio(Vec<f32>),
    StartContinuous,
    StopContinuous,
    Shutdown,
}

pub enum EngineResult {
    ModelReady,
    ModelError(String),
    TranscriptionDone(String),
    TranscriptionChunk(String),
    TranscriptionError(String),
    ContinuousStarted,
    ContinuousStopped,
}

pub struct Engine {
    cmd_tx: Sender<EngineCommand>,
    _thread: std::thread::JoinHandle<()>,
}

impl Engine {
    pub fn new(model_dir: String) -> (Self, Receiver<EngineResult>) {
        let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded();
        let (result_tx, result_rx) = crossbeam_channel::unbounded();
        let thread = std::thread::Builder::new()
            .name("engine".to_string())
            .spawn(move || {
                engine_thread(model_dir, cmd_rx, result_tx);
            })
            .expect("Failed to spawn engine thread");

        (
            Self {
                cmd_tx,
                _thread: thread,
            },
            result_rx,
        )
    }

    pub fn send(&self, cmd: EngineCommand) {
        let _ = self.cmd_tx.send(cmd);
    }

    pub fn cmd_tx(&self) -> Sender<EngineCommand> {
        self.cmd_tx.clone()
    }

    pub fn shutdown(&self) {
        let _ = self.cmd_tx.send(EngineCommand::Shutdown);
    }
}


fn drain_tokens(stream: *mut std::os::raw::c_void) -> String {
    // Collect raw tokens (preserving boundaries for space-fixing)
    let mut tokens: Vec<String> = Vec::new();
    let mut ptrs: [*const c_char; 64] = [std::ptr::null(); 64];
    loop {
        let n = unsafe { ffi::vox_stream_get(stream, ptrs.as_mut_ptr(), 64) };
        if n <= 0 {
            break;
        }
        for i in 0..n as usize {
            if !ptrs[i].is_null() {
                if let Ok(s) = unsafe { CStr::from_ptr(ptrs[i]) }.to_str() {
                    tokens.push(s.to_string());
                }
            }
        }
    }

    if tokens.is_empty() {
        return String::new();
    }

    // Join naively first to check if we're in "spaceless mode"
    let naive: String = tokens.iter().map(|s| s.as_str()).collect();
    let alpha_count = naive.chars().filter(|c| c.is_alphabetic()).count();
    let space_count = naive.chars().filter(|c| *c == ' ').count();

    // Spaceless mode: substantial Latin text with almost no spaces.
    // This happens when the model starts with CJK tokens — BPE tokens for
    // subsequent Latin text lose their leading space character.
    let spaceless = alpha_count > 20 && space_count * 20 < alpha_count;

    if !spaceless {
        // Normal mode: only fix CJK ↔ Latin/digit transitions
        return fix_cjk_spacing(&naive);
    }

    // Spaceless mode: re-join tokens, inserting spaces at token boundaries
    let mut result = String::new();
    for token in &tokens {
        if let (Some(prev), Some(first)) = (result.chars().last(), token.chars().next()) {
            if !prev.is_whitespace() && !first.is_whitespace() {
                if (is_cjk(prev) && first.is_ascii_alphanumeric())
                    || (prev.is_ascii_alphanumeric() && is_cjk(first))
                {
                    // CJK ↔ Latin boundary
                    result.push(' ');
                } else if prev.is_ascii_alphabetic() && first.is_ascii_alphabetic() {
                    // Latin word boundary without space — the core fix
                    result.push(' ');
                }
            }
        }
        result.push_str(token);
    }

    result
}

/// Insert spaces at CJK ↔ Latin/digit transitions (character-level).
/// Always safe — Chinese/Japanese/Korean never have inter-character spaces,
/// so a space at the boundary with Latin text is always correct.
fn fix_cjk_spacing(text: &str) -> String {
    let mut result = String::with_capacity(text.len() + 16);
    let mut prev: Option<char> = None;
    for ch in text.chars() {
        if let Some(p) = prev {
            if !p.is_whitespace() && !ch.is_whitespace() {
                if (is_cjk(p) && ch.is_ascii_alphanumeric())
                    || (p.is_ascii_alphanumeric() && is_cjk(ch))
                {
                    result.push(' ');
                }
            }
        }
        result.push(ch);
        prev = Some(ch);
    }
    result
}

fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}'   // CJK Unified Ideographs
        | '\u{3400}'..='\u{4DBF}' // CJK Extension A
        | '\u{F900}'..='\u{FAFF}' // CJK Compatibility Ideographs
        | '\u{3000}'..='\u{303F}' // CJK Symbols and Punctuation
        | '\u{3040}'..='\u{309F}' // Hiragana
        | '\u{30A0}'..='\u{30FF}' // Katakana
        | '\u{AC00}'..='\u{D7AF}' // Hangul Syllables
    )
}

fn engine_thread(
    model_dir: String,
    cmd_rx: Receiver<EngineCommand>,
    result_tx: Sender<EngineResult>,
) {
    // Redirect C stderr to a log file so we can see voxtral.c error messages
    let stderr_log = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.join("voxtral_stderr.log")))
        .unwrap_or_else(|| std::path::PathBuf::from("voxtral_stderr.log"));
    if let Ok(path) = CString::new(stderr_log.to_string_lossy().as_ref()) {
        let mode = CString::new("w").unwrap();
        unsafe {
            let stderr_stream = __acrt_iob_func(2); // stderr
            freopen(path.as_ptr(), mode.as_ptr(), stderr_stream);
        }
        info!("C stderr redirected to {}", stderr_log.display());
    }

    // Enable full CUDA pipeline (encoder + decoder + adapter on GPU)
    // Must use CRT's _putenv_s so C code's getenv() sees it (std::env::set_var
    // uses Win32 SetEnvironmentVariable which doesn't update the CRT copy)
    unsafe {
        let name = CString::new("VOX_CUDA_FAST").unwrap();
        let value = CString::new("1").unwrap();
        _putenv_s(name.as_ptr(), value.as_ptr());
    }

    // Enable verbose output from voxtral.c
    unsafe { ffi::vox_verbose = 2; }

    info!("Loading model from: {}", model_dir);

    let c_model_dir = match CString::new(model_dir.clone()) {
        Ok(s) => s,
        Err(e) => {
            let _ = result_tx.send(EngineResult::ModelError(format!(
                "Invalid model dir path: {}",
                e
            )));
            return;
        }
    };

    let ctx = unsafe { ffi::vox_load(c_model_dir.as_ptr()) };
    if ctx.is_null() {
        let _ = result_tx.send(EngineResult::ModelError(
            "vox_load failed".to_string(),
        ));
        return;
    }

    let cuda_ok = unsafe { ffi::vox_cuda_available() };
    if cuda_ok != 0 {
        let name = unsafe { CStr::from_ptr(ffi::vox_cuda_device_name()) }
            .to_str()
            .unwrap_or("unknown");
        info!("CUDA available: {}", name);
    } else {
        info!("WARNING: CUDA not available, running on CPU (will be slow)");
    }

    // Pre-warm encoder dequant + GPU cache so first transcription is fast
    if cuda_ok != 0 {
        info!("Warming up encoder cache...");
        unsafe { ffi::vox_cuda_warmup_encoder(ctx) };
        info!("Encoder cache warm");
    }

    info!("Model loaded successfully");
    let _ = result_tx.send(EngineResult::ModelReady);

    // Persistent stream state (for continuous mode)
    let mut stream: Option<*mut std::os::raw::c_void> = None;

    loop {
        match cmd_rx.recv() {
            // Push-to-talk: independent transcription (KV reset each time)
            Ok(EngineCommand::Transcribe(samples)) => {
                info!("Transcribing {} samples ({:.1}s)", samples.len(), samples.len() as f64 / 16000.0);

                let result_ptr = unsafe {
                    ffi::vox_transcribe_audio(ctx, samples.as_ptr(), samples.len() as i32)
                };

                if result_ptr.is_null() {
                    let _ = result_tx.send(EngineResult::TranscriptionError(
                        "vox_transcribe_audio returned null".to_string(),
                    ));
                    continue;
                }

                let text = unsafe { CStr::from_ptr(result_ptr) }
                    .to_str()
                    .unwrap_or("")
                    .trim()
                    .to_string();

                unsafe { ffi::free(result_ptr as *mut _) };

                let text = fix_cjk_spacing(&text);

                info!("Transcription: '{}'", text);
                let _ = result_tx.send(EngineResult::TranscriptionDone(text));
            }

            // Mode 3: start continuous streaming
            Ok(EngineCommand::StartContinuous) => {
                // Free any existing stream
                if let Some(s) = stream.take() {
                    unsafe { ffi::vox_stream_free(s) };
                }

                let s = unsafe { ffi::vox_stream_init(ctx) };
                if s.is_null() {
                    let _ = result_tx.send(EngineResult::TranscriptionError(
                        "vox_stream_init failed for continuous mode".to_string(),
                    ));
                    continue;
                }

                // Lower processing interval for responsive streaming
                unsafe { ffi::vox_set_processing_interval(s, 0.5) };

                stream = Some(s);
                info!("Continuous streaming started");
                let _ = result_tx.send(EngineResult::ContinuousStarted);
            }

            // Mode 3: feed audio chunk directly (no VAD)
            Ok(EngineCommand::FeedAudio(samples)) => {
                let s = match stream {
                    Some(s) => s,
                    None => continue,
                };

                // Feed all audio straight to the encoder — let the model's
                // [P] padding tokens handle silence naturally
                unsafe {
                    ffi::vox_stream_feed(s, samples.as_ptr(), samples.len() as i32);
                }

                // Drain any available tokens
                let text = drain_tokens(s);
                if !text.is_empty() {
                    let _ = result_tx.send(EngineResult::TranscriptionChunk(text));
                }
            }

            // Mode 3: stop continuous streaming
            Ok(EngineCommand::StopContinuous) => {
                if let Some(s) = stream.take() {
                    // Finish the stream to get remaining tokens
                    unsafe { ffi::vox_stream_finish(s) };
                    let text = drain_tokens(s);
                    if !text.is_empty() {
                        let _ = result_tx.send(EngineResult::TranscriptionChunk(text));
                    }
                    unsafe { ffi::vox_stream_free(s) };
                    info!("Continuous streaming stopped");
                }
                let _ = result_tx.send(EngineResult::ContinuousStopped);
            }

            Ok(EngineCommand::Shutdown) | Err(_) => {
                info!("Engine shutting down");
                break;
            }
        }
    }

    // Cleanup
    if let Some(s) = stream {
        unsafe { ffi::vox_stream_free(s) };
    }
    unsafe { ffi::vox_free(ctx) };
}
