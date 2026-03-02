#![allow(non_camel_case_types, dead_code)]

use std::os::raw::{c_char, c_float, c_int, c_void};

// Opaque types
pub enum LlamaModel {}
pub enum LlamaContext {}
pub enum LlamaSampler {}
pub enum LlamaVocab {}
pub enum MtmdContext {}
pub enum MtmdBitmap {}
pub enum MtmdImageTokens {}
pub enum MtmdInputChunk {}
pub enum MtmdInputChunks {}

// llama_memory_t is a pointer to an opaque type
pub type LlamaMemoryT = *mut c_void;

pub type LlamaToken = i32;
pub type LlamaPos = i32;
pub type LlamaSeqId = i32;

// Enums
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum LlamaFlashAttnType {
    Auto = -1,
    Disabled = 0,
    Enabled = 1,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum LlamaSplitMode {
    None = 0,
    Layer = 1,
    Row = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum LlamaRopeScalingType {
    Unspecified = -1,
    None = 0,
    Linear = 1,
    Yarn = 2,
    LongRope = 3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum LlamaPoolingType {
    Unspecified = -1,
    None = 0,
    Mean = 1,
    Cls = 2,
    Last = 3,
    Rank = 4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum LlamaAttentionType {
    Unspecified = -1,
    Causal = 0,
    NonCausal = 1,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum MtmdInputChunkType {
    Text = 0,
    Image = 1,
    Audio = 2,
}

// ggml_type enum (we only need the default values)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // ... many more, but we just use defaults
    Count = 39,
}

// Structs
#[repr(C)]
pub struct LlamaModelParams {
    pub devices: *mut c_void,
    pub tensor_buft_overrides: *const c_void,
    pub n_gpu_layers: i32,
    pub split_mode: LlamaSplitMode,
    pub main_gpu: i32,
    pub tensor_split: *const c_float,
    pub progress_callback: Option<extern "C" fn(c_float, *mut c_void) -> bool>,
    pub progress_callback_user_data: *mut c_void,
    pub kv_overrides: *const c_void,
    pub vocab_only: bool,
    pub use_mmap: bool,
    pub use_direct_io: bool,
    pub use_mlock: bool,
    pub check_tensors: bool,
    pub use_extra_bufts: bool,
    pub no_host: bool,
    pub no_alloc: bool,
}

#[repr(C)]
pub struct LlamaSamplerSeqConfig {
    pub seq_id: LlamaSeqId,
    pub sampler: *mut LlamaSampler,
}

#[repr(C)]
pub struct LlamaContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub rope_scaling_type: LlamaRopeScalingType,
    pub pooling_type: LlamaPoolingType,
    pub attention_type: LlamaAttentionType,
    pub flash_attn_type: LlamaFlashAttnType,
    pub rope_freq_base: c_float,
    pub rope_freq_scale: c_float,
    pub yarn_ext_factor: c_float,
    pub yarn_attn_factor: c_float,
    pub yarn_beta_fast: c_float,
    pub yarn_beta_slow: c_float,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: c_float,
    pub cb_eval: Option<extern "C" fn(*mut c_void, bool) -> bool>,
    pub cb_eval_user_data: *mut c_void,
    pub type_k: GgmlType,
    pub type_v: GgmlType,
    pub abort_callback: Option<extern "C" fn(*mut c_void) -> bool>,
    pub abort_callback_data: *mut c_void,
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
    pub samplers: *mut LlamaSamplerSeqConfig,
    pub n_samplers: usize,
}

#[repr(C)]
pub struct LlamaSamplerChainParams {
    pub no_perf: bool,
}

#[repr(C)]
pub struct LlamaBatch {
    pub n_tokens: i32,
    pub token: *mut LlamaToken,
    pub embd: *mut c_float,
    pub pos: *mut LlamaPos,
    pub n_seq_id: *mut i32,
    pub seq_id: *mut *mut LlamaSeqId,
    pub logits: *mut i8,
}

#[repr(C)]
pub struct MtmdContextParams {
    pub use_gpu: bool,
    pub print_timings: bool,
    pub n_threads: c_int,
    pub image_marker: *const c_char,
    pub media_marker: *const c_char,
    pub flash_attn_type: LlamaFlashAttnType,
    pub warmup: bool,
    pub image_min_tokens: c_int,
    pub image_max_tokens: c_int,
    pub cb_eval: Option<extern "C" fn(*mut c_void, bool) -> bool>,
    pub cb_eval_user_data: *mut c_void,
}

#[repr(C)]
pub struct MtmdInputText {
    pub text: *const c_char,
    pub add_special: bool,
    pub parse_special: bool,
}

extern "C" {
    // Backend init/free
    pub fn llama_backend_init();
    pub fn llama_backend_free();

    // Log control
    pub fn llama_log_set(
        log_callback: Option<extern "C" fn(c_int, *const c_char, *mut c_void)>,
        user_data: *mut c_void,
    );

    // Model
    pub fn llama_model_default_params() -> LlamaModelParams;
    pub fn llama_model_load_from_file(
        path_model: *const c_char,
        params: LlamaModelParams,
    ) -> *mut LlamaModel;
    pub fn llama_model_free(model: *mut LlamaModel);
    pub fn llama_model_get_vocab(model: *const LlamaModel) -> *const LlamaVocab;

    // Context
    pub fn llama_context_default_params() -> LlamaContextParams;
    pub fn llama_init_from_model(
        model: *mut LlamaModel,
        params: LlamaContextParams,
    ) -> *mut LlamaContext;
    pub fn llama_free(ctx: *mut LlamaContext);
    pub fn llama_get_model(ctx: *const LlamaContext) -> *const LlamaModel;
    pub fn llama_get_memory(ctx: *const LlamaContext) -> LlamaMemoryT;

    // Memory
    pub fn llama_memory_clear(mem: LlamaMemoryT, data: bool);

    // Batch
    pub fn llama_batch_get_one(
        tokens: *mut LlamaToken,
        n_tokens: i32,
    ) -> LlamaBatch;
    pub fn llama_batch_init(n_tokens: i32, embd: i32, n_seq_max: i32) -> LlamaBatch;
    pub fn llama_batch_free(batch: LlamaBatch);

    // Decode
    pub fn llama_decode(ctx: *mut LlamaContext, batch: LlamaBatch) -> i32;

    // Sampling
    pub fn llama_sampler_chain_default_params() -> LlamaSamplerChainParams;
    pub fn llama_sampler_chain_init(
        params: LlamaSamplerChainParams,
    ) -> *mut LlamaSampler;
    pub fn llama_sampler_chain_add(chain: *mut LlamaSampler, smpl: *mut LlamaSampler);
    pub fn llama_sampler_init_greedy() -> *mut LlamaSampler;
    pub fn llama_sampler_sample(
        smpl: *mut LlamaSampler,
        ctx: *mut LlamaContext,
        idx: i32,
    ) -> LlamaToken;
    pub fn llama_sampler_free(smpl: *mut LlamaSampler);

    // Token conversion
    pub fn llama_token_to_piece(
        vocab: *const LlamaVocab,
        token: LlamaToken,
        buf: *mut c_char,
        length: i32,
        lstrip: i32,
        special: bool,
    ) -> i32;
    pub fn llama_vocab_is_eog(vocab: *const LlamaVocab, token: LlamaToken) -> bool;

    // MTMD
    pub fn mtmd_default_marker() -> *const c_char;
    pub fn mtmd_context_params_default() -> MtmdContextParams;
    pub fn mtmd_init_from_file(
        mmproj_fname: *const c_char,
        text_model: *const LlamaModel,
        ctx_params: MtmdContextParams,
    ) -> *mut MtmdContext;
    pub fn mtmd_free(ctx: *mut MtmdContext);
    pub fn mtmd_support_audio(ctx: *mut MtmdContext) -> bool;
    pub fn mtmd_get_audio_bitrate(ctx: *mut MtmdContext) -> c_int;

    // Bitmap
    pub fn mtmd_bitmap_init_from_audio(
        n_samples: usize,
        data: *const c_float,
    ) -> *mut MtmdBitmap;
    pub fn mtmd_bitmap_free(bitmap: *mut MtmdBitmap);

    // Input chunks
    pub fn mtmd_input_chunks_init() -> *mut MtmdInputChunks;
    pub fn mtmd_input_chunks_free(chunks: *mut MtmdInputChunks);

    // Tokenize
    pub fn mtmd_tokenize(
        ctx: *mut MtmdContext,
        output: *mut MtmdInputChunks,
        text: *const MtmdInputText,
        bitmaps: *const *const MtmdBitmap,
        n_bitmaps: usize,
    ) -> i32;

    // Helper eval
    pub fn mtmd_helper_eval_chunks(
        ctx: *mut MtmdContext,
        lctx: *mut LlamaContext,
        chunks: *const MtmdInputChunks,
        n_past: LlamaPos,
        seq_id: LlamaSeqId,
        n_batch: i32,
        logits_last: bool,
        new_n_past: *mut LlamaPos,
    ) -> i32;

    // Helper bitmap from buffer (fallback)
    pub fn mtmd_helper_bitmap_init_from_buf(
        ctx: *mut MtmdContext,
        buf: *const u8,
        len: usize,
    ) -> *mut MtmdBitmap;
}
