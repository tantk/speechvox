# Voxtral 4B Inference Pipeline

This document describes the complete inference pipeline of the Voxtral 4B Realtime speech-to-text model: from raw audio input through mel spectrogram extraction, audio encoding, adapter downsampling, and decoder token generation.

---

## 1. Pipeline Overview

```
Raw Audio (16kHz mono PCM)
    ↓
Mel Spectrogram [n_frames, 128]
    ↓
Conv Stem (stride 2): [n_frames, 128] → [n_frames/2, 1280]
    ↓
Audio Encoder (32 transformer layers): [n_frames/2, 1280] → [n_frames/2, 1280]
    ↓
Adapter (4x downsample + projection): [n_frames/2, 1280] → [n_frames/8, 3072]
    ↓
Decoder Prefill (26 transformer layers): KV cache initialized
    ↓
Autoregressive Generation: adapter_embed + token_embed → next token
    ↓
Tekken BPE Tokenizer: token IDs → text
```

---

## 2. Model Architecture Constants

```c
/* Audio preprocessing */
#define VOX_SAMPLE_RATE      16000     /* Input sample rate */
#define VOX_MEL_BINS         128       /* Mel filterbank bins */
#define VOX_HOP_LENGTH       160       /* 10ms per frame */
#define VOX_WINDOW_SIZE      400       /* 25ms window */

/* Audio encoder */
#define VOX_ENC_DIM          1280      /* Embedding dimension */
#define VOX_ENC_LAYERS       32        /* Transformer layers */
#define VOX_ENC_HEADS        32        /* Attention heads (full MHA) */
#define VOX_ENC_KV_HEADS     32        /* KV heads = Q heads */
#define VOX_ENC_HEAD_DIM     64        /* 1280 / 32 = 64 (custom) */
#define VOX_ENC_HIDDEN       5120      /* FFN hidden dim (4x) */
#define VOX_ENC_WINDOW       750       /* Causal sliding window */

/* Downsampling */
#define VOX_DOWNSAMPLE       4         /* Adapter downsample factor */

/* LLM decoder */
#define VOX_DEC_DIM          3072      /* Embedding dimension */
#define VOX_DEC_LAYERS       26        /* Transformer layers */
#define VOX_DEC_HEADS        32        /* Query heads */
#define VOX_DEC_KV_HEADS     8         /* KV heads (GQA 4:1 ratio) */
#define VOX_DEC_HEAD_DIM     128       /* Per-head dimension */
#define VOX_DEC_HIDDEN       9216      /* FFN hidden dim (3x) */
#define VOX_DEC_WINDOW       8192      /* KV sliding window */
#define VOX_VOCAB_SIZE       131072    /* Tekken BPE vocabulary */
#define VOX_ROPE_THETA       1000000.0 /* RoPE frequency base */
```

---

## 3. Audio Preprocessing: Mel Spectrogram

**File:** `voxtral_audio.c`

### Parameters

- Sample rate: 16 kHz
- FFT size: 400 (N_FREQ = 201 bins)
- Window: 400-sample Hann window (25ms)
- Hop: 160 samples (10ms)
- Mel bins: 128 (Slaney-style triangular filterbank)
- Padding: reflect, 200 samples on each side (center=True STFT convention)

### Processing Steps

1. **Reflect padding**: 200 samples on left and right
2. **Windowed STFT**: Sliding 400-sample Hann window with 160-sample stride, direct DFT to produce power spectrum
3. **Mel filterbank**: 128 triangular filters (Slaney auditory scale) applied to 201 frequency bins
4. **Log scaling**: `log10(mel)`, clamped to `[LOG_MEL_MAX - 8, LOG_MEL_MAX]` where `LOG_MEL_MAX = 1.5`
5. **Normalization**: `(log_mel + 4) / 4`

Output: `[n_frames, 128]` float32, where `n_frames ≈ n_samples / 160`.

### Streaming Mel API

```c
vox_mel_ctx_t *vox_mel_ctx_init(int left_pad_samples);
int vox_mel_feed(vox_mel_ctx_t *ctx, const float *samples, int n_samples);
int vox_mel_finish(vox_mel_ctx_t *ctx, int right_pad_samples);
float *vox_mel_data(vox_mel_ctx_t *ctx, int *out_n_frames);
void vox_mel_free(vox_mel_ctx_t *ctx);
```

Frames are computed incrementally as soon as 160 new samples arrive. `mel_finish()` applies right-side reflect padding for proper boundary handling.

---

## 4. Audio Encoder

**File:** `voxtral_encoder.c`

### Conv Stem (Causal, Stride-2)

The encoder begins with two 1D causal convolutions that transform mel features into the encoder's embedding space:

```
mel [n_frames, 128]
    ↓ transpose → [128, n_frames]
    ↓ Conv0: kernel=3, stride=1, channels 128→1280, GELU activation
    ↓ Conv1: kernel=3, stride=2, channels 1280→1280, GELU activation
    ↓ transpose → [n_frames/2, 1280]
```

The stride-2 conv halves the sequence length.

### 32 Transformer Layers

Each layer follows the standard pre-norm transformer pattern:

```
For each layer (0..31):
    1. RMSNorm(x, eps=1e-5)
    2. Multi-Head Attention:
       - Q = norm(x) @ Wq [1280→2048] + bias_q     (32 heads × 64 dim)
       - K = norm(x) @ Wk [1280→2048]               (32 heads × 64 dim, no bias)
       - V = norm(x) @ Wv [1280→2048] + bias_v      (32 heads × 64 dim)
       - RoPE applied to Q, K (theta=1e6)
       - Causal attention with sliding window=750, scale=1/sqrt(64)
       - Output = softmax(QK^T / scale) @ V
       - Projection = attn_out @ Wo [2048→1280] + bias_o
    3. Residual: x += projection
    4. RMSNorm(x, eps=1e-5)
    5. SwiGLU FFN:
       - gate = SiLU(norm(x) @ W1 [1280→5120])      (no bias)
       - up   = norm(x) @ W3 [1280→5120]             (no bias)
       - down  = (gate * up) @ W2 [5120→1280] + bias (bias on down-proj only)
    6. Residual: x += down
Final: RMSNorm(x)
```

Key: The encoder uses **full MHA** (32 Q heads, 32 KV heads) unlike the decoder which uses GQA.

### Incremental Encoding

For streaming, only new mel frames are processed:

```c
float *vox_encoder_forward_incremental(vox_ctx_t *ctx,
                                       const float *x_new,
                                       int new_len,
                                       int *out_len);
```

Per-layer KV caches (`enc_kv_cache_k/v [layers × max_pos × 2048]`) store previous keys/values. When the cache exceeds `VOX_ENC_WINDOW` (750), it compacts by discarding old entries and shifting positions.

---

## 5. Audio-Language Adapter

**File:** `voxtral_encoder.c` (tail section)

The adapter bridges the encoder (1280-dim audio) to the decoder (3072-dim language):

```
Encoder output [seq_len, 1280]
    ↓ 4x downsample: concatenate 4 consecutive vectors
    ↓ [seq_len/4, 5120]
    ↓ Linear: [5120→3072], no bias, BF16 weights
    ↓ GELU activation
    ↓ Linear: [3072→3072], no bias, BF16 weights
    ↓ [seq_len/4, 3072]
```

The 4x downsample is critical: it reduces the token rate from ~50 tokens/second (encoder) to ~12.5 tokens/second (decoder), matching the model's `VOX_FRAME_RATE = 12.5`.

**Minimum decoder prompt**: 39 adapter tokens (≈312 mel frames ≈ 3 seconds of audio).

---

## 6. LLM Decoder

**File:** `voxtral_decoder.c`

### Dual-Stream Input (Delayed Streams Modeling)

Unlike standard LLMs, the decoder receives **two embeddings per position** that are summed:

```
input_embeds[pos] = adapter_embed[pos] + tok_embed[prev_token]
```

- `adapter_embed`: Output from the audio adapter (audio context)
- `tok_embed`: Embedding of the previously generated token (language context)

This dual-stream design is core to Voxtral 4B Realtime's architecture. The decoder fuses audio and language information at every step.

### Decoder Architecture (26 Layers with GQA)

```
For each layer (0..25):
    1. RMSNorm(x, eps=1e-5) × (1 + ada_scale[layer])    ← Adaptive timing
    2. Grouped Query Attention (GQA):
       - Q = norm(x) @ Wq [3072→4096]    (32 heads × 128 dim)
       - K = norm(x) @ Wk [3072→1024]    (8 KV heads × 128 dim)
       - V = norm(x) @ Wv [3072→1024]    (8 KV heads × 128 dim)
       - RoPE applied to Q, K (theta=1e6)
       - Q heads 0-3 share KV head 0, Q heads 4-7 share KV head 1, etc.
       - Causal attention with sliding window=8192, scale=1/sqrt(128)
       - Projection = attn_out @ Wo [4096→3072]
    3. Residual: x += projection
    4. RMSNorm(x, eps=1e-5) × (1 + ada_scale[layer])    ← Adaptive timing
    5. SwiGLU FFN:
       - gate = SiLU(norm(x) @ W1 [3072→9216])
       - up   = norm(x) @ W3 [3072→9216]
       - down  = (gate * up) @ W2 [9216→3072]
    6. Residual: x += down
Final: RMSNorm(x) → logits = x @ tok_embeddings^T [3072→131072]
```

### Adaptive Timing Conditioning

Each layer's RMSNorm output is scaled by a time-dependent factor:

```c
// Precomputed at model load (or when delay changes):
t_value = delay_tokens;  // default 6 (480ms)
t_cond = sinusoidal_embedding(t_value, dim=3072);
ada_scale[layer] = ada_up @ GELU(ada_down @ t_cond);
// ada_down: [32, 3072], ada_up: [3072, 32] (bottleneck MLP)

// At inference:
x_norm = rms_norm(x) * (1 + ada_scale[layer]);
```

This conditions the decoder on the expected transcription delay, improving quality at the configured latency.

### Grouped Query Attention (GQA)

GQA reduces KV cache size by sharing KV heads across query heads:

- 32 query heads, 8 KV heads → 4:1 sharing ratio
- Q heads [0,1,2,3] attend to KV head 0
- Q heads [4,5,6,7] attend to KV head 1
- ...and so on

KV cache per layer: `8 × 128 = 1024` dims (vs 4096 for full MHA), saving 4x memory.

### KV Cache & Sliding Window

```c
// KV cache: [26 layers × 8192 positions × 1024 dims]
// Total: ~400 MB (FP32) or ~200 MB (FP16 on GPU)

// When kv_cache_len > 8192:
static void kv_cache_compact(vox_ctx_t *ctx) {
    int keep = VOX_DEC_WINDOW;       // 8192
    int discard = kv_cache_len - keep;
    // Shift: cache[0:keep] = cache[discard:discard+keep]
    // Update kv_pos_offset += discard (for correct RoPE positions)
}
```

The sliding window caps memory usage regardless of audio length.

---

## 7. Token Generation

### Special Tokens

```c
#define TOKEN_BOS           1       // Beginning of sequence
#define TOKEN_EOS           2       // End of sequence / end of transcription
#define TOKEN_STREAMING_PAD 32      // Padding for streaming delay
```

### Decoder Prompt Construction

Before autoregressive generation, the decoder is seeded with:

```
[BOS] + [STREAMING_PAD × (32 + delay_tokens)] + adapter_tokens
```

Where `delay_tokens` defaults to 6 (480ms at 80ms/token).

### Generation Loop

```c
while (gen_pos < total_adapter && !eos_seen) {
    // 1. Compute input embedding (dual-stream fusion)
    step_embed = adapter_buf[gen_pos] + tok_embeddings[prev_token];

    // 2. Decoder forward pass (single token, uses KV cache)
    int token = vox_decoder_forward(ctx, step_embed, logits);

    // 3. Check for EOS
    if (token == TOKEN_EOS) { eos_seen = 1; break; }

    // 4. Decode token to text and queue
    const char *text = vox_tokenizer_decode(tokenizer, token);
    enqueue_token(text);

    prev_token = token;
    gen_pos++;
}
```

Each token corresponds to ~80ms of audio (`RAW_AUDIO_LENGTH_PER_TOK = 1280 samples`).

### Tekken BPE Tokenizer

- Vocabulary: 131,072 tokens
- Loaded from `tekken.json` in the model directory
- Byte-level BPE with Unicode support
- Token decode returns a `const char*` pointer into the tokenizer's internal string table

---

## 8. Streaming API

**File:** `voxtral.c`

### Stream State

```c
struct vox_stream {
    vox_ctx_t *ctx;
    vox_tokenizer_t *tokenizer;

    /* Mel computation */
    vox_mel_ctx_t *mel_ctx;
    int mel_cursor;             // How many mel frames have been encoded

    /* Conv stem boundary state */
    float *mel_tail;            // [128 × 2] last 2 frames
    float *conv0_tail;          // [1280 × 2] conv0 boundary
    float *conv0_residual;      // [1280] pending output
    int enc_residual_count;     // 0-3 (adapter alignment buffer)

    /* Adapter output (growing buffer) */
    float *adapter_buf;         // [total_adapter, 3072]
    int total_adapter;

    /* Decoder state */
    int decoder_started;
    int gen_pos;                // Next adapter position for generation
    int prev_token;             // Last generated token
    int eos_seen;
    int finished;

    /* Token output queue */
    const char **token_queue;
    int queue_head, queue_tail;

    /* Timing */
    int min_new_mel;            // Encoder trigger threshold
};
```

### API Functions

| Function | Purpose |
|----------|---------|
| `vox_stream_init(ctx)` | Create stream, allocate buffers |
| `vox_stream_feed(s, samples, n)` | Feed PCM, compute mel, trigger encoder+decoder when ready |
| `vox_stream_flush(s)` | Zero-pad to trigger encoder on remaining mel frames |
| `vox_stream_finish(s)` | Finalize: mel_finish + final pass, permanently closes stream |
| `vox_stream_end_utterance(s)` | Like finish but keeps stream alive for next utterance |
| `vox_stream_get(s, out, max)` | Drain decoded text tokens from queue |
| `vox_stream_reset_encoder(s)` | Reset encoder KV but preserve decoder KV (split cache trick) |
| `vox_stream_free(s)` | Free all stream resources |
| `vox_set_processing_interval(s, sec)` | Set minimum mel frames before encoder triggers |

### Internal Processing Flow

```
vox_stream_feed(samples)
    ↓ vox_mel_feed() → new mel frames
    ↓ Check: n_new_mel >= min_new_mel?
        Yes → stream_run_encoder()
                ↓ Conv stem (incremental, boundary-aware)
                ↓ Encoder forward (incremental, KV cached)
                ↓ Adapter forward (4x downsample)
                ↓ Append to adapter_buf
              stream_run_decoder()
                ↓ If !decoder_started && total_adapter >= 39:
                    Prefill decoder with all adapter tokens
                ↓ Generate tokens for new adapter positions
                ↓ Enqueue token strings
        No → return (accumulate more audio)
```

### Processing Interval

```c
void vox_set_processing_interval(vox_stream_t *s, float seconds) {
    s->min_new_mel = (int)(seconds * VOX_SAMPLE_RATE / VOX_HOP_LENGTH);
}
// Default: 2.0s → 200 mel frames
// Low-latency: 0.5s → 50 mel frames
```

---

## 9. Multi-Utterance Mode (Split Cache Trick)

For sequential utterances within a session, the encoder KV is reset while the decoder KV is preserved:

```c
int vox_stream_reset_encoder(vox_stream_t *s) {
    // Reset: encoder KV cache, mel context, conv stem state, adapter buffer
    // Preserve: decoder KV cache, gen_pos, prev_token, decoder_started
    // Clear: finished, eos_seen
}
```

This gives the decoder semantic context from previous utterances (e.g., names mentioned earlier) while starting fresh audio encoding.

### End-of-Utterance Handling

`vox_stream_end_utterance()` captures the last word that `flush()` alone misses:

```c
int vox_stream_end_utterance(vox_stream_t *s) {
    int mel_before = s->mel_cursor;

    // 1. Flush pending audio
    vox_stream_flush(s);

    // 2. Finalize mel (right-side padding)
    s->finished = 1;
    vox_mel_finish(s->mel_ctx, 0);

    // 3. Re-encode (CUDA chunked encoder may re-encode from scratch)
    stream_run_encoder(s);

    // 4. Adapter relabeling: compute how many tokens are genuinely new
    int re_encode_tokens = s->total_adapter;
    int old_tokens = mel_before / (2 * VOX_DOWNSAMPLE);  // conv stride 2 × adapter 4x
    int new_tokens = re_encode_tokens - old_tokens;

    // 5. Adjust adapter state so decoder only sees new tokens
    s->adapter_pos_offset = s->gen_pos - old_tokens;
    s->total_adapter = s->gen_pos + new_tokens;
    vox_cuda_stream_adapter_relabel(s->ctx, s->adapter_pos_offset);

    // 6. Final decoder pass to generate remaining tokens
    stream_run_decoder(s);

    // 7. Keep stream alive
    s->finished = 0;
    s->eos_seen = 0;
}
```

---

## 10. CUDA Acceleration

**File:** `voxtral_cuda.c`

### Execution Modes

1. **CUDA Full Pipeline** (`VOX_CUDA_FAST=1`): Entire encoder and decoder run on GPU. Intermediate tensors stay on device. CUDA Graphs eliminate per-token CPU-GPU sync overhead.

2. **CUDA-Assisted**: CPU handles control flow, GPU accelerates individual matmuls via cuBLAS and custom kernels.

3. **CPU Fallback**: All computation on CPU (automatic if CUDA unavailable).

### Key CUDA Operations

| Kernel | Purpose |
|--------|---------|
| `vox_cuda_encode_adapter_stream_append` | Full encoder+adapter on GPU, append to device-side adapter buffer |
| `vox_cuda_decoder_forward_full` | Single-token decoder step, KV on device |
| `vox_cuda_decoder_prefill_full` | Multi-token prefill on GPU |
| `vox_cuda_causal_attention` | Generic causal attention (multiple kernel versions v2-v6) |
| `vox_cuda_attention_step` | Single-query attention with device KV cache |
| `vox_cuda_quant_matmul_t` | Quantized GEMV on GPU |
| `vox_cuda_linear_bf16` | BF16 weight matmul via cuBLAS |
| `vox_cuda_linear2_bf16` | Fused dual linear projection (decoder hot path) |

### Attention Kernel Evolution

The attention implementation has been iteratively optimized:

- **v2**: Vectorized loads/stores (default fallback if v3+ unavailable)
- **v3**: Chunked processing (256-token blocks) for large KV sequences
- **v4**: Fused KV append into attention kernel
- **v5**: Skip inactive chunks (empty KV regions)
- **v6**: Store partial results in FP16 for memory efficiency

V2 is enabled by default as the fallback path when V3+ kernels fail to load from the cubin. This ensures vectorized attention even on untested GPU/driver combinations. Override with `VOX_CUDA_ATTN_V2=0` to force the scalar V1 path.

### Quantized GEMV Kernel Optimizations

The GEMV kernels (`vox_gemv_q8_0`, `vox_gemv_q4_0`, `vox_gemv_q4_k` + beta variants) use two key optimizations:

**Vectorized weight loads**: Instead of reading quantized weights one byte at a time, each kernel uses `uint4` (16-byte) loads to fetch entire quant blocks in 1-2 memory transactions:
- Q8_0: 2 × uint4 loads for 32 int8 quants (was 32 byte reads)
- Q4_0: 1 × uint4 load for 16 packed nibble bytes (was 16 byte reads)
- Q4_K: 1 × uint4 load per sub-block × 8 sub-blocks (was 128 byte reads)

Values are unpacked from `uint32_t` fields in registers using bit shifts and masks.

**Shared memory bank conflict padding**: The activation vector in shared memory is padded by 1 float per 32-element row (`SH_PAD(idx) = idx + idx/32`). Without padding, all 32 warp lanes access the same bank on every loop iteration (32-way conflict, fully serialized). With padding, each lane hits a different bank (zero conflicts). Shared memory allocation is increased accordingly: `(K + K/32) * sizeof(float)`.

### GPU KV Cache

```c
// Optional FP16 KV cache (halves VRAM, default on)
// Controlled by VOX_CUDA_KV_FP16 environment variable

void vox_cuda_kv_cache_compact(ctx, discard, keep, kv_dim, max_seq);
void vox_cuda_kv_cache_reset(ctx);
int  vox_cuda_kv_cache_download_host(ctx, start_pos, n_pos);  // For CPU fallback
```

### Device-Side Adapter Buffer

In CUDA full pipeline mode, adapter embeddings stay on GPU:

```c
void vox_cuda_stream_adapter_reset(ctx);       // Clear buffer
void vox_cuda_stream_adapter_set_offset(ctx, offset);   // Set position (destructive)
void vox_cuda_stream_adapter_relabel(ctx, new_offset);   // Relabel position (preserves data)
void vox_cuda_stream_adapter_compact(ctx, consumed);     // Ring-buffer discard
int  vox_cuda_stream_adapter_copy_prompt(ctx, out, n);   // Download for CPU prefill
```

### VRAM Usage

Approximate VRAM breakdown (Q4_K quantized, RTX 4070 Ti):

| Component | Size |
|-----------|------|
| Encoder weights (32 layers, quantized) | ~600 MB |
| Decoder weights (26 layers, quantized) | ~500 MB |
| Adapter + embeddings (BF16) | ~800 MB |
| KV caches (FP16) | ~200 MB |
| Working buffers | ~50 MB |
| **Total** | **~2.15 GB** |

---

## 11. Rust Integration

**File:** `src/engine.rs`

### Three Transcription Modes

```rust
pub enum EngineCommand {
    Transcribe(Vec<f32>),                // Mode 1: Independent (KV reset)
    TranscribeKeepSession(Vec<f32>),     // Mode 2: Multi-utterance (persistent KV)
    ResetSession,                        // Mode 2: Manual session clear
    FeedAudio(Vec<f32>),                 // Mode 3: Continuous streaming chunks
    StartContinuous,                     // Mode 3: Begin listening
    StopContinuous,                      // Mode 3: Stop listening
    Shutdown,
}
```

### Mode 1: Push-to-Talk (Independent)

Uses `vox_transcribe_audio()` which internally creates/destroys a stream per call. Full KV cache reset between utterances.

### Mode 2: Multi-Utterance Session

Maintains a persistent `vox_stream_t*` across utterances:

```rust
// First utterance: init + feed + end_utterance + drain
// Subsequent: reset_encoder + feed + end_utterance + drain
// Session timeout: free stream after configurable minutes of inactivity
```

### Mode 3: Continuous Streaming

Always-listening with Voice Activity Detection (VAD):

```rust
const VAD_WINDOW: usize = 160;          // 10ms at 16kHz
const VAD_SILENCE_THRESH: f32 = 0.002;  // RMS energy threshold
const VAD_SILENCE_PASS: usize = 60;     // 600ms natural gap passthrough

// For each 10ms window:
//   rms = sqrt(sum(samples^2) / 160)
//   if rms > 0.002: voice → feed to stream
//   if rms ≤ 0.002 for ≤600ms: pass through (natural word gap)
//   if rms ≤ 0.002 for >600ms: flush + skip (silence)
```

### Token Drain Helper

```rust
fn drain_tokens(stream: *mut c_void) -> String {
    let mut result = String::new();
    let mut ptrs: [*const c_char; 64] = [std::ptr::null(); 64];
    loop {
        let n = unsafe { ffi::vox_stream_get(stream, ptrs.as_mut_ptr(), 64) };
        if n <= 0 { break; }
        for i in 0..n as usize {
            if let Ok(s) = unsafe { CStr::from_ptr(ptrs[i]) }.to_str() {
                result.push_str(s);
            }
        }
    }
    result
}
```

---

## 12. End-to-End Example

### Input: 10 seconds of speech at 16 kHz

```
160,000 PCM samples
    ↓ Mel spectrogram
~1000 mel frames [1000, 128]
    ↓ Conv stem (stride 2)
500 encoder tokens [500, 1280]
    ↓ 32 encoder layers
500 encoded tokens [500, 1280]
    ↓ Adapter (4x downsample)
125 adapter tokens [125, 3072]
    ↓ Decoder prefill (26 layers)
KV cache initialized with 125 positions
    ↓ Autoregressive generation
~125 decoder steps → ~125 text tokens
    ↓ Tekken tokenizer decode
"hello this is a test of the speech recognition system"
```

### Timing (RTX 4070 Ti, CUDA Full, Q4_K)

| Stage | Time |
|-------|------|
| Mel spectrogram | ~10 ms |
| Encoder (32 layers) | ~120 ms |
| Adapter | ~3 ms |
| Decoder prefill | ~20 ms |
| Token generation (125 tokens) | ~190 ms |
| **Total** | **~340 ms** |

Real-time factor: 0.034x (29x faster than real-time).

### Streaming Latency

With `processing_interval = 0.5s` and `delay_tokens = 6`:
- First token appears after ~3 seconds (minimum 39 adapter tokens)
- Subsequent tokens: ~1.5 ms each
- End-to-end latency: ~500ms after sufficient audio buffered
