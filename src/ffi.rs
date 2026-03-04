#![allow(dead_code)]

use std::os::raw::{c_char, c_float, c_int, c_void};

extern "C" {
    // Core model management
    pub fn vox_load(model_dir: *const c_char) -> *mut c_void;
    pub fn vox_free(ctx: *mut c_void);
    pub fn vox_set_delay(ctx: *mut c_void, delay_ms: c_int);

    // Batch transcription (returns malloc'd string, caller must free)
    pub fn vox_transcribe_audio(
        ctx: *mut c_void,
        samples: *const c_float,
        n_samples: c_int,
    ) -> *mut c_char;

    // CUDA availability check
    pub fn vox_cuda_available() -> c_int;
    pub fn vox_cuda_device_name() -> *const c_char;

    // Pre-populate encoder dequant + GPU BF16 cache at load time
    pub fn vox_cuda_warmup_encoder(ctx: *mut c_void);

    // Streaming API (for multi-utterance and continuous modes)
    pub fn vox_stream_init(ctx: *mut c_void) -> *mut c_void;
    pub fn vox_stream_feed(s: *mut c_void, samples: *const c_float, n: c_int) -> c_int;
    pub fn vox_stream_flush(s: *mut c_void) -> c_int;
    pub fn vox_stream_finish(s: *mut c_void) -> c_int;
    pub fn vox_stream_end_utterance(s: *mut c_void) -> c_int;
    pub fn vox_stream_get(s: *mut c_void, out: *mut *const c_char, max: c_int) -> c_int;
    pub fn vox_stream_free(s: *mut c_void);
    pub fn vox_stream_reset_encoder(s: *mut c_void) -> c_int;
    pub fn vox_set_processing_interval(s: *mut c_void, seconds: c_float);

    // Verbose control
    pub static mut vox_verbose: c_int;

    // C stdlib free (to release strings returned by vox_transcribe_audio)
    pub fn free(ptr: *mut c_void);
}
