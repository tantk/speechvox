use crate::ffi;
use anyhow::Result;
use crossbeam_channel::{Receiver, Sender};
use std::ffi::{CStr, CString};
use tracing::{debug, error, info};

pub enum EngineCommand {
    Transcribe(Vec<f32>),
    Shutdown,
}

pub enum EngineResult {
    ModelReady,
    ModelError(String),
    TranscriptionDone(String),
    TranscriptionError(String),
}

pub struct Engine {
    cmd_tx: Sender<EngineCommand>,
    _thread: std::thread::JoinHandle<()>,
}

impl Engine {
    pub fn new(
        model_path: String,
        mmproj_path: String,
        use_gpu: bool,
    ) -> (Self, Receiver<EngineResult>) {
        let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded();
        let (result_tx, result_rx) = crossbeam_channel::unbounded();
        let thread = std::thread::Builder::new()
            .name("engine".to_string())
            .spawn(move || {
                engine_thread(model_path, mmproj_path, use_gpu, cmd_rx, result_tx);
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

    pub fn transcribe(&self, audio: Vec<f32>) {
        let _ = self.cmd_tx.send(EngineCommand::Transcribe(audio));
    }

    pub fn shutdown(&self) {
        let _ = self.cmd_tx.send(EngineCommand::Shutdown);
    }
}

fn engine_thread(
    model_path: String,
    mmproj_path: String,
    use_gpu: bool,
    cmd_rx: Receiver<EngineCommand>,
    result_tx: Sender<EngineResult>,
) {
    unsafe {
        // Initialize backend
        ffi::llama_backend_init();

        // Suppress llama.cpp logs (they go to stderr by default)
        ffi::llama_log_set(Some(llama_log_callback), std::ptr::null_mut());

        // Load model
        info!("Loading model: {}", model_path);
        let mut model_params = ffi::llama_model_default_params();
        model_params.n_gpu_layers = if use_gpu { -1 } else { 0 };

        let c_model_path = match CString::new(model_path.clone()) {
            Ok(s) => s,
            Err(e) => {
                let _ = result_tx.send(EngineResult::ModelError(format!(
                    "Invalid model path: {}",
                    e
                )));
                return;
            }
        };

        let model = ffi::llama_model_load_from_file(c_model_path.as_ptr(), model_params);
        if model.is_null() {
            let _ = result_tx.send(EngineResult::ModelError(
                "Failed to load model".to_string(),
            ));
            return;
        }

        // Create llama context
        let mut ctx_params = ffi::llama_context_default_params();
        ctx_params.n_ctx = 8192;
        ctx_params.n_batch = 2048;
        ctx_params.n_ubatch = 512;
        ctx_params.no_perf = true;

        let ctx = ffi::llama_init_from_model(model, ctx_params);
        if ctx.is_null() {
            let _ = result_tx.send(EngineResult::ModelError(
                "Failed to create llama context".to_string(),
            ));
            ffi::llama_model_free(model);
            return;
        }

        // Initialize mtmd context
        info!("Loading mmproj: {}", mmproj_path);
        let mut mtmd_params = ffi::mtmd_context_params_default();
        mtmd_params.use_gpu = use_gpu;
        mtmd_params.n_threads = 4;

        let c_mmproj_path = match CString::new(mmproj_path.clone()) {
            Ok(s) => s,
            Err(e) => {
                let _ = result_tx.send(EngineResult::ModelError(format!(
                    "Invalid mmproj path: {}",
                    e
                )));
                ffi::llama_free(ctx);
                ffi::llama_model_free(model);
                return;
            }
        };

        let mtmd_ctx =
            ffi::mtmd_init_from_file(c_mmproj_path.as_ptr(), model as *const _, mtmd_params);
        if mtmd_ctx.is_null() {
            let _ = result_tx.send(EngineResult::ModelError(
                "Failed to initialize mtmd context".to_string(),
            ));
            ffi::llama_free(ctx);
            ffi::llama_model_free(model);
            return;
        }

        // Check audio support
        if !ffi::mtmd_support_audio(mtmd_ctx) {
            let _ = result_tx.send(EngineResult::ModelError(
                "Model does not support audio input".to_string(),
            ));
            ffi::mtmd_free(mtmd_ctx);
            ffi::llama_free(ctx);
            ffi::llama_model_free(model);
            return;
        }

        let bitrate = ffi::mtmd_get_audio_bitrate(mtmd_ctx);
        info!("Audio bitrate: {} Hz", bitrate);

        // Create sampler (greedy for deterministic transcription)
        let sampler_params = ffi::llama_sampler_chain_default_params();
        let sampler = ffi::llama_sampler_chain_init(sampler_params);
        let greedy = ffi::llama_sampler_init_greedy();
        ffi::llama_sampler_chain_add(sampler, greedy);

        info!("Model loaded successfully");
        let _ = result_tx.send(EngineResult::ModelReady);

        // Get vocab for token conversion
        let vocab = ffi::llama_model_get_vocab(model as *const _);

        // Process commands
        loop {
            match cmd_rx.recv() {
                Ok(EngineCommand::Transcribe(samples)) => {
                    debug!("Transcribing {} samples", samples.len());
                    match transcribe(
                        mtmd_ctx, ctx, model, sampler, vocab, &samples,
                    ) {
                        Ok(text) => {
                            let _ = result_tx.send(EngineResult::TranscriptionDone(text));
                        }
                        Err(e) => {
                            let _ = result_tx.send(EngineResult::TranscriptionError(format!(
                                "{}",
                                e
                            )));
                        }
                    }
                }
                Ok(EngineCommand::Shutdown) | Err(_) => {
                    info!("Engine shutting down");
                    break;
                }
            }
        }

        // Cleanup
        ffi::llama_sampler_free(sampler);
        ffi::mtmd_free(mtmd_ctx);
        ffi::llama_free(ctx);
        ffi::llama_model_free(model);
        ffi::llama_backend_free();
    }
}

unsafe fn transcribe(
    mtmd_ctx: *mut ffi::MtmdContext,
    ctx: *mut ffi::LlamaContext,
    _model: *mut ffi::LlamaModel,
    sampler: *mut ffi::LlamaSampler,
    vocab: *const ffi::LlamaVocab,
    samples: &[f32],
) -> Result<String> {
    // Clear KV cache before each transcription
    let mem = ffi::llama_get_memory(ctx as *const _);
    ffi::llama_memory_clear(mem, false);

    // Create audio bitmap from PCM samples
    let bitmap = ffi::mtmd_bitmap_init_from_audio(samples.len(), samples.as_ptr());
    if bitmap.is_null() {
        return Err(anyhow::anyhow!("Failed to create audio bitmap"));
    }

    // Build prompt with media marker
    let marker = ffi::mtmd_default_marker();
    let marker_str = CStr::from_ptr(marker).to_str().unwrap_or("<__media__>");
    let prompt = format!("{}Transcribe this audio.", marker_str);
    let c_prompt = CString::new(prompt)?;

    let input_text = ffi::MtmdInputText {
        text: c_prompt.as_ptr(),
        add_special: true,
        parse_special: true,
    };

    // Tokenize
    let chunks = ffi::mtmd_input_chunks_init();
    let bitmap_ptr: *const ffi::MtmdBitmap = bitmap;
    let res = ffi::mtmd_tokenize(mtmd_ctx, chunks, &input_text, &bitmap_ptr, 1);

    ffi::mtmd_bitmap_free(bitmap);

    if res != 0 {
        ffi::mtmd_input_chunks_free(chunks);
        return Err(anyhow::anyhow!("mtmd_tokenize failed with code {}", res));
    }

    // Evaluate chunks
    let mut n_past: ffi::LlamaPos = 0;
    let eval_res = ffi::mtmd_helper_eval_chunks(
        mtmd_ctx,
        ctx,
        chunks as *const _,
        n_past,
        0,    // seq_id
        2048, // n_batch
        true, // logits_last
        &mut n_past,
    );

    ffi::mtmd_input_chunks_free(chunks);

    if eval_res != 0 {
        return Err(anyhow::anyhow!(
            "mtmd_helper_eval_chunks failed with code {}",
            eval_res
        ));
    }

    // Generate tokens
    let mut result = String::new();
    let max_tokens = 2048;
    let mut buf = [0i8; 256];

    for _ in 0..max_tokens {
        let token = ffi::llama_sampler_sample(sampler, ctx, -1);

        if ffi::llama_vocab_is_eog(vocab, token) {
            break;
        }

        // Convert token to text
        let n = ffi::llama_token_to_piece(
            vocab,
            token,
            buf.as_mut_ptr(),
            buf.len() as i32,
            0,
            false,
        );

        if n > 0 {
            let piece = std::str::from_utf8(std::slice::from_raw_parts(
                buf.as_ptr() as *const u8,
                n as usize,
            ))
            .unwrap_or("");
            result.push_str(piece);
        }

        // Decode next token
        let mut token_mut = token;
        let batch = ffi::llama_batch_get_one(&mut token_mut, 1);
        let decode_res = ffi::llama_decode(ctx, batch);
        if decode_res != 0 {
            error!("llama_decode failed during generation: {}", decode_res);
            break;
        }
        let _ = n_past;
    }

    debug!("Transcription result: '{}'", result.trim());
    Ok(result.trim().to_string())
}

extern "C" fn llama_log_callback(
    level: std::os::raw::c_int,
    text: *const std::os::raw::c_char,
    _user_data: *mut std::os::raw::c_void,
) {
    if text.is_null() {
        return;
    }
    let msg = unsafe { CStr::from_ptr(text) }
        .to_str()
        .unwrap_or("")
        .trim();
    if msg.is_empty() {
        return;
    }
    match level {
        2 => error!("[llama] {}", msg),   // GGML_LOG_LEVEL_ERROR
        3 => debug!("[llama] {}", msg),   // GGML_LOG_LEVEL_WARN
        _ => debug!("[llama] {}", msg),   // INFO and DEBUG
    }
}
