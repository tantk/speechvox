---
pipeline_tag: automatic-speech-recognition
language:
  - en
  - es
  - fr
  - de
  - it
  - pt
  - nl
  - ru
  - ar
  - hi
  - ko
  - ja
  - zh
license: apache-2.0
tags:
  - automatic-speech-recognition
  - multilingual
  - quantized
  - voxtral
  - real-time
  - cuda
base_model: mistralai/Voxtral-Mini-4B-Realtime-2602
base_model_relation: quantized
metrics:
  - wer
  - cer
inference: false
---

# Voxtral 4B Realtime — VQF Quantized

Pre-quantized weights for [Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) — Mistral AI's 4.4B streaming speech-to-text model. Packaged in the **VQF** (Voxtral Quantized Format) container, inspired by [voxtral.c](https://github.com/HorizonXP/voxtral.c).

At the default 480ms delay, Voxtral matches Whisper large-v3; at 960ms+ it surpasses it. These quantizations reduce VRAM from ~8.8 GB (BF16) down to 2-5 GB with minimal quality loss.

| File | Quant | Size | Bits/Wt | VRAM | Use Case |
|------|-------|------|---------|------|----------|
| `consolidated-q4_k.vqf` | Q4_K | 3.0 GB | 4.6 | ~4 GB | **Recommended.** Best quality-to-size ratio. |
| `consolidated-q4_0.vqf` | Q4_0 | 3.2 GB | 4.0 | ~4 GB | Simplest format, fastest dequant. |
| `consolidated-q6_k.vqf` | Q6_K | 3.8 GB | 6.4 | ~5 GB | Near-lossless. Sweet spot for 8 GB GPUs. |
| `consolidated-q8_0.vqf` | Q8_0 | 5.0 GB | 9.0 | ~7 GB | Reference quality. 12+ GB GPUs. |

VRAM includes ~1.5 GB overhead for KV caches and working buffers. Also includes the Tekken BPE tokenizer (`tekken.json`).

## Quickstart

These weights work with [voxtral.c](https://github.com/HorizonXP/voxtral.c) or any runtime supporting the VQF format. Requires an NVIDIA GPU with CUDA 12.0+.

## What's Quantized

406 linear weight matrices across the 32 encoder + 26 decoder layers (wq/wk/wv/wo/w1/w2/w3). Everything else stays BF16: embeddings, norms, biases, adapter weights, conv stem.

## Benchmarks

### WER by Language (BF16, FLEURS, 480ms)

From the [original paper](https://arxiv.org/abs/2602.11298). Quantized variants track within ~0.1-0.3% WER.

| | EN | ES | FR | DE | IT | PT | NL | RU | AR | HI | KO | JA | ZH |
|-|----|----|----|----|----|----|----|----|----|----|----|----|-----|
| WER/CER | 4.9 | 3.3 | 6.4 | 6.2 | 5.7 | 5.4 | 8.4 | 5.4 | 14.4 | 12.9 | 11.4 | 9.6* | 10.5* |

\* CER for Japanese/Chinese.

### Delay vs. Quality

| Delay | FLEURS WER | vs. Whisper large-v3 |
|-------|-----------|---------------------|
| 480ms | 8.72% | Matches (8.23%) |
| 960ms | 7.70% | Surpasses |
| 2400ms | 6.73% | Surpasses |

### Speed (RTX 4070 Ti, Q4_K, 10s audio)

~340 ms total (**29x real-time**): encoder 120ms + decoder 190ms + overhead 30ms.

## Architecture

4.4B parameter causal encoder-decoder: 32-layer audio encoder (1280D, full MHA) -> 4x adapter downsample -> 26-layer language decoder (3072D, GQA 32Q/8KV) -> Tekken tokenizer (131K vocab).

<details>
<summary>Detailed architecture & constants</summary>

```
16kHz mono PCM -> Mel (128 bins) -> Conv Stem (stride 2, 50Hz)
  -> Encoder: 32 layers, 1280D, 32 heads, 64 head-dim, SwiGLU 5120, RoPE, window 750
  -> Adapter: concat 4 frames (5120) -> Linear -> GELU -> Linear (3072), 12.5Hz
  -> Decoder: 26 layers, 3072D, 32Q/8KV heads, 128 head-dim, SwiGLU 9216, RoPE, window 8192
  -> Tekken BPE (131K vocab)
```

| | Encoder | Decoder |
|--|---------|---------|
| Dim | 1280 | 3072 |
| Layers | 32 | 26 |
| Q / KV heads | 32 / 32 | 32 / 8 |
| Head dim | 64 | 128 |
| FFN hidden | 5120 | 9216 |
| Window | 750 | 8192 |

</details>

## VQF Format

Memory-mapped binary container: `[VQF1 header] [tensor descriptors] [64B-aligned data]`. Weights are read directly from the mapped file without copying.

<details>
<summary>Block format details</summary>

**Q4_0** (20B / 32 values) — `float scale` + `uint8 nibs[16]`. Symmetric: `val = scale * (nibble - 8)`

**Q4_K** (148B / 256 values) — `float super_scale, super_min` + `uint8 scales[12]` (6-bit packed) + `uint8 nibs[128]`. Asymmetric with per-sub-block scale+min.

**Q6_K** (204B / 256 values) — `float super_scale` + `int8 scales[8]` + `uint8 ql[128]` + `uint8 qh[64]`. Symmetric: 6-bit = 4-bit low + 2-bit high. `val = super_scale * scales[sub] * (q6 - 32)`

**Q8_0** (36B / 32 values) — `float scale` + `int8 quants[32]`. Symmetric: `val = scale * quant`

</details>

## How to Quantize

```bash
pip install torch safetensors

# Q4_K (recommended)
python quantize/quantize.py /path/to/model_dir consolidated-q4_k.vqf --type Q4_K

# Also: --type Q4_0 | Q6_K | Q8_0
```

~15 seconds per variant on an RTX 4070 Ti. Requires PyTorch with CUDA.

## Limitations

- Transcription only (no translation, no diarization, no timestamps)
- May hallucinate on non-speech audio — use with a VAD in production
- Accent/dialect coverage varies; noisy environments reduce accuracy
- Q4_K recommended over Q4_0 for low-resource languages

## Links

- [Original model](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) | [Paper](https://arxiv.org/abs/2602.11298) | [voxtral.c](https://github.com/HorizonXP/voxtral.c)

## Citation

```bibtex
@misc{voxtral2025realtime,
    title={Voxtral Realtime},
    author={Alexander H. Liu and Andy Ehrenberg and Andy Lo and Angad Kalra and Anna Googasian and Barret Zoph and Bilal Piot and Changil Kim and Daniel Haziza and Daphne Ippolito and David Grangier and Edouard Grave and Francisco Massa and Guillaume Lample and Jade Copet and Leo Boytsov and Luca Wehrstedt and Martin Sundermeyer and Marta R. Costa-jussà and Michael Auli and Mona Diab and Patrick von Platen and Paul-Ambroise Duquenne and Robin Algayres and Ruslan Mavlyutov and Sravya Popuri and Timothée Lacroix and Vineel Pratap},
    year={2025},
    eprint={2602.11298},
    archivePrefix={arXiv}
}
```
