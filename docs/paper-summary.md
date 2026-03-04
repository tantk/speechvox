# Voxtral Realtime: Paper Summary & Feature Comparison

**Paper:** "Voxtral Realtime" (arXiv:2602.11298)
**Authors:** Alexander H. Liu, Andy Ehrenberg, Andy Lo, et al. (Mistral AI)
**Model:** Voxtral-Mini-4B-Realtime-2602, Apache 2.0 license

---

## 1. What Is Voxtral Realtime?

A natively streaming automatic speech recognition (ASR) model that matches offline transcription quality (Whisper-level) at sub-second latency. Unlike conventional ASR systems that process audio in fixed chunks then transcribe, Voxtral Realtime uses causal attention throughout so it can transcribe audio as it arrives, with configurable latency from 80ms to 2.4 seconds.

Key claim: at 480ms delay, it matches Whisper (the most widely deployed offline ASR) in quality. At 960ms, it surpasses Whisper. At 2400ms, it approaches offline-only models like Voxtral Mini Transcribe V2.

---

## 2. Architecture

4.4B parameters total, three components:

```
Audio (16kHz PCM)
    |
    v
[Mel Spectrogram] 128 bins, 10ms hop (160 samples)
    |
    v
[Causal Conv Stem] 2x stride → 50 Hz output
    |
    v
[Audio Encoder] 32 layers, 1280D, 970M params
    |  - Full MHA (32 Q heads, 32 KV heads, 64 head dim)
    |  - RMSNorm, SwiGLU FFN (5120 hidden), RoPE
    |  - Causal sliding window: 750 frames (15 seconds)
    |
    v
[Adapter MLP] 4x temporal downsample → 12.5 Hz
    |  - Concatenate 4 frames: [4 x 1280] = 5120
    |  - Linear 5120 → 3072, GELU, Linear 3072 → 3072
    |  - 25M params
    |
    v
[Language Decoder] 26 layers, 3072D, 3.4B params
    |  - GQA: 32 Q heads, 8 KV heads, 128 head dim
    |  - SwiGLU FFN (9216 hidden), RoPE
    |  - Sliding window: 8192 tokens
    |  - Initialized from Ministral 3B
    |  - Ada RMS-Norm for delay conditioning
    |
    v
[Tekken Tokenizer] 131K vocabulary → text
```

### Key Design Choices vs. Whisper

| Aspect | Whisper | Voxtral Realtime |
|--------|---------|-----------------|
| Encoder attention | Bidirectional | Causal |
| Normalization | LayerNorm | RMSNorm |
| FFN activation | GELU | SwiGLU |
| Positional encoding | Sinusoidal | RoPE |
| Encoder window | Fixed (30s chunks) | Sliding (750 frames, 15s) |
| Streaming | No (chunk-and-stitch) | Native |
| Decoder KV heads | Full MHA | GQA (8 KV / 32 Q) |

---

## 3. Delayed Streams Modeling

The core innovation. At each decoder step, the input is a **sum of two embeddings**:

```
input_embeds[t] = adapter_embed[t] + tok_embed[prev_token[t]]
```

- **Audio stream** (adapter_embed): What the model hears at time t
- **Text stream** (tok_embed): What the model said at time t-1

This dual-stream architecture lets the decoder simultaneously attend to audio context and maintain language coherence.

### Token Types

Each decoder step (80ms of audio) produces exactly one token:

| Token | Meaning |
|-------|---------|
| **[P]** (padding) | No word to emit yet — model is waiting for acoustic evidence |
| **[W]** (word boundary) | A word has been fully heard AND the configured delay has elapsed — begin emitting text |
| **BPE tokens** | Actual text subword pieces (token IDs >= 1000) |

Example at 480ms delay:

```
Audio:     "Hello world"
Time(ms):  0    80   160  240  320  400  480  560  640  720
Tokens:    [P]  [P]  [P]  [P]  [P]  [P]  [W]  Hello [W] world
                                          ^--- 480ms after "Hello" starts
```

### Word Grouping

When consecutive words fall within the same 80ms frame, [W] is NOT inserted between them. The subword tokens follow directly. This preserves the BPE tokenizer's learned word distributions — the decoder sees the same token sequences it learned during language model pretraining.

---

## 4. Ada RMS-Norm (Delay Conditioning)

A single model serves ALL delays (80ms–2400ms) via adaptive normalization. In each decoder layer:

```
r_attn = Attention(RMSNorm(x))
h = x + r_attn

r_ffn = FFN(RMSNorm(h) * (1 + g(τ)))    ← delay conditioning here
y = h + r_ffn
```

Where `g(τ)` is computed once at session start:
1. Sinusoidal time embedding of delay τ → 3072-dim vector
2. Bottleneck MLP: Linear(3072 → 32), GELU, Linear(32 → 3072)
3. Result: per-layer scale vector that modulates the FFN normalization

Total overhead: ~5M extra parameters (negligible for a 4.4B model).

The attention branch is NOT conditioned — only the FFN is. This was found to converge faster and produce lower WER than alternatives (embedding addition, special tokens).

---

## 5. Training

### Three Phases

1. **Encoder warm-up (5% of training)**: Freeze decoder, train encoder + adapter only. Prevents randomly-initialized encoder from destabilizing the pretrained Ministral 3B decoder.

2. **End-to-end (95% of training)**: Train all components jointly. LR drops from 4e-4 to 6e-5.

3. **Optimization**: AdamW, batch size 370 hours of audio.

### Key Training Techniques

- **Delay sampling**: Each training sample uses a random delay τ ∈ {80, 160, ..., 2400}ms, sampled uniformly. This teaches the model to operate at any delay.

- **z-loss**: Penalty on logit norm magnitude. Without this, text embedding norms grow while audio embedding norms shrink, causing the model to ignore audio entirely.

- **Left-padding**: 32 frames of zero audio + [P] tokens prepended to every input. Acts as an "attention sink" — improves WER by ~0.6% on English.

### Dataset

Large-scale dataset spanning 13 languages. Specific composition not detailed in the paper.

---

## 6. Supported Languages

13 languages with reported benchmarks:

| Language | FLEURS WER (480ms) |
|----------|--------------------|
| English | 4.90% |
| Spanish | 3.31% |
| French | 6.42% |
| German | 6.19% |
| Italian | 5.65% |
| Portuguese | 5.38% |
| Dutch | 8.38% |
| Russian | 5.43% |
| Arabic | 14.38% |
| Hindi | 12.91% |
| Korean | 11.38% |
| Japanese | 9.59% CER |
| Chinese | 10.45% CER |

---

## 7. Benchmark Results

### Macro-Average WER (%)

| Model | Delay | En-Short | En-Long | FLEURS | MCV |
|-------|-------|----------|---------|--------|-----|
| Whisper large-v3 | offline | 8.39 | 7.97 | 8.23 | 14.25 |
| Voxtral Mini V2 | offline | 7.27 | 7.11 | 5.90 | 8.07 |
| **Voxtral Realtime** | **480ms** | **8.47** | **7.73** | **8.72** | **15.24** |
| **Voxtral Realtime** | **960ms** | **7.94** | **7.13** | **7.70** | **11.99** |
| **Voxtral Realtime** | **2400ms** | **7.72** | **6.93** | **6.73** | **10.47** |

Key takeaway: 480ms delay ≈ Whisper quality. 960ms+ surpasses Whisper.

---

## 8. Streaming Inference

### vLLM Integration (Reference Implementation)

- Separate KV caches for encoder (50Hz) and decoder (12.5Hz)
- Custom attention backend stretches encoder KV block size by p=4 for unified paged allocation
- Resumable requests: persistent KV blocks reused across audio chunks
- Full-duplex: ingest next 80ms while decoding current token
- WebSocket API for bidirectional streaming

### Memory Bounds

- Encoder sliding window: 750 frames → bounded at 15 seconds of context
- Decoder sliding window: 8192 tokens → bounded at ~10 minutes of generation
- Both enable "infinite streaming" with constant memory

---

## 9. What the Paper Does NOT Cover

The paper focuses on the model architecture and training. It explicitly does NOT describe:

- Voice Activity Detection (VAD) — paper states the model works "without external VAD"
- Multi-utterance sessions
- Quantization or model compression
- Speaker diarization
- Timestamps / word-level alignment (used in training targets, not exposed at inference)
- Translation (transcription only)
- Language detection / forcing
- Code-switching handling
- On-device deployment details

---

## 10. Feature Comparison: Paper vs. SpeechVox

### Capabilities from the Paper

| Feature | Paper Describes | SpeechVox Implements | Notes |
|---------|:-:|:-:|-------|
| Causal audio encoder (32 layers) | Y | Y | voxtral.c encoder |
| Conv stem (stride 2, causal) | Y | Y | voxtral_encoder.c |
| 4x adapter downsample | Y | Y | voxtral_encoder.c |
| GQA decoder (26 layers) | Y | Y | voxtral_decoder.c |
| Dual-stream input (adapter + tok_embed) | Y | Y | Core of voxtral.c generation loop |
| Ada RMS-Norm (delay conditioning) | Y | Y | vox_set_delay() in FFI |
| [P] padding tokens | Y | Y | TOKEN_STREAMING_PAD = 32, filtered at output |
| [W] word boundary tokens | Y | Y | Handled internally, filtered (token ID < 1000) |
| Left-padding (32 frames) | Y | Y | Prompt: BOS + 32 STREAMING_PAD + delay tokens |
| Configurable delay (80–2400ms) | Y | Partial | `vox_set_delay()` exists but not exposed in UI |
| Sliding window encoder (750 frames) | Y | Y | KV cache compaction in encoder |
| Sliding window decoder (8192 tokens) | Y | Y | KV cache compaction in decoder |
| Infinite streaming | Y | Y | Bounded memory via sliding windows |
| 13 languages | Y | Y | Model capability; no per-language UI |
| Tekken tokenizer (131K vocab) | Y | Y | voxtral_tokenizer.c |

### Enhancements Beyond the Paper

| Feature | Paper | SpeechVox | Notes |
|---------|:-----:|:---------:|-------|
| Push-to-Talk mode | - | Y | Mode 1: independent utterances, KV reset |
| Multi-utterance sessions | - | Y | Mode 2: split cache trick, persistent decoder KV |
| Continuous streaming with VAD | - | Y | Mode 3: RMS-based VAD, always-listening |
| VQF quantization (Q4_K/Q4_0/Q8_0) | - | Y | Custom format, GPU-accelerated dequant |
| CUDA full pipeline | - | Y | Encoder + decoder on GPU, CUDA Graphs |
| System tray UI | - | Y | Mode selection, session reset |
| Overlay status indicator | - | Y | Recording/processing/listening states |
| Push-to-talk hotkey (F9) | - | Y | Low-level keyboard hook |
| CJK↔Latin space fixing | - | Y | Post-processing for cross-language tokenization |
| End-of-utterance handling | - | Y | vox_stream_end_utterance() for last-word capture |

### Features NOT in Either (Potential Additions)

| Feature | Status | Difficulty | Value |
|---------|--------|------------|-------|
| Configurable delay UI | Missing | Low | Expose delay slider in tray menu (80–2400ms) |
| Language selection UI | Missing | Low | Let user force a language or show detected language |
| Word-level timestamps | Not supported by model | N/A | Model generates [W] tokens but doesn't expose timestamps |
| Speaker diarization | Not supported by model | N/A | Would need a separate model |
| Translation | Not supported by model | N/A | Model is transcription-only |
| Punctuation control | Implicit | Low | Model generates punctuation naturally; could add toggle |
| WebSocket API | Paper's vLLM has it | Medium | Could expose a local WebSocket endpoint |

### Gap Analysis

**Things we could add easily:**

1. **Delay configuration UI**: The C backend already supports `vox_set_delay(ctx, delay_ms)`. Just needs a slider or dropdown in the tray menu (80, 160, 240, 320, 400, 480, 640, 960, 1200, 2400ms). Lower delay = faster but less accurate. Default 480ms is the sweet spot.

2. **Language indicator**: The model auto-detects language. We could display the detected language in the overlay (by checking the first few generated tokens against known Unicode ranges).

**Things we can't add (model limitation):**

1. **Timestamps**: The model uses [W] word-boundary tokens during training to learn timing, but at inference it only generates text tokens. There's no timestamp output.

2. **Diarization**: Single-speaker model. Would need a separate speaker embedding model.

3. **Translation**: The model was trained for transcription only (repeat pattern). The original Voxtral (non-realtime) supports translation via the continuation pattern, but the Realtime variant does not.

---

## 11. Summary

SpeechVox implements the complete Voxtral 4B Realtime inference pipeline as described in the paper, plus several significant enhancements:

- **Three transcription modes** (push-to-talk, multi-utterance, continuous) vs. the paper's single streaming mode
- **Custom quantization** (VQF format with Q4_K) reducing VRAM from ~8.8GB to ~2.15GB
- **Full CUDA pipeline** with custom kernels and CUDA Graphs
- **Desktop integration** (system tray, hotkey, overlay, keyboard simulation)

The main gap is **delay configuration in the UI** — the model's marquee feature (operating at any delay from 80ms to 2.4s) is supported by the backend but not exposed to the user. This would be a high-value, low-effort addition.

Sources:
- [Voxtral Realtime paper (arXiv:2602.11298)](https://arxiv.org/abs/2602.11298)
- [HuggingFace model card](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
- [Mistral AI announcement](https://mistral.ai/news/voxtral)
