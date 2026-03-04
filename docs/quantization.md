# Voxtral 4B Model Quantization

This document describes how the Voxtral 4B Realtime model weights are quantized from their original safetensors format into the custom VQF (Voxtral Quantized Format) binary container, and how quantized weights are loaded and dequantized at runtime.

---

## 1. VQF File Format

VQF is a custom binary container designed for memory-mapped access to quantized tensors.

### Magic & Version

```c
#define VQF_MAGIC   0x31465156   /* "VQF1" little-endian */
#define VQF_VERSION 1
```

### File Layout

```
[VQF Header          (24 bytes)]
[Tensor Descriptor 0 (variable)]
[Tensor Descriptor 1 (variable)]
...
[Tensor Descriptor N-1          ]
[Zero padding to 64-byte alignment]
[Tensor Data Blocks ...]
```

### Header (24 bytes)

```c
typedef struct {
    uint32_t magic;          /* 0x31465156 */
    uint32_t version;        /* 1 */
    uint32_t default_qtype;  /* Default quant type for linear layers */
    uint32_t num_tensors;    /* Number of tensor descriptors */
    uint64_t data_offset;    /* Byte offset to tensor data (64-byte aligned) */
} vqf_header_t;
```

### Tensor Descriptor (variable length)

```c
typedef struct {
    uint16_t name_len;
    char     name[256];          /* UTF-8 tensor name */
    uint32_t qtype;              /* VQF_TYPE_* enum */
    uint32_t ndim;               /* Number of dimensions (max 4) */
    int64_t  shape[4];           /* Shape array */
    uint64_t data_offset;        /* Relative to header's data_offset */
    uint64_t data_size;          /* Bytes */
} vqf_tensor_desc_t;
```

### Quantization Type Constants

| Type       | ID  | Description                       | Bytes/Element |
|------------|-----|-----------------------------------|---------------|
| `VQF_TYPE_F32`  | 0   | No quantization                  | 4.00          |
| `VQF_TYPE_F16`  | 1   | Half-precision float             | 2.00          |
| `VQF_TYPE_BF16` | 2   | BFloat16                         | 2.00          |
| `VQF_TYPE_Q8_0` | 8   | 8-bit, 32 values/block           | 1.125         |
| `VQF_TYPE_Q4_0` | 10  | 4-bit, 32 values/block           | 0.625         |
| `VQF_TYPE_Q4_K` | 12  | 4-bit with sub-block scales, 256 values/super-block | 0.578 |

---

## 2. Block Quantization Structures

All block formats follow GGML conventions: a fixed-size block containing a scale factor and packed quantized values.

### Q8_0 (32 values/block, 36 bytes)

```c
typedef struct {
    float  scale;        /* 4 bytes: per-block scale factor */
    int8_t quants[32];   /* 32 bytes: signed 8-bit quantized values */
} vqf_block_q8_0;        /* Total: 36 bytes */
```

**Math:**
- Quantize: `scale = max(|values|) / 127`, `quant[i] = clamp(round(value[i] / scale), -127, 127)`
- Dequantize: `value[i] = scale * quant[i]`

### Q4_0 (32 values/block, 20 bytes)

```c
typedef struct {
    float   scale;       /* 4 bytes: per-block scale factor */
    uint8_t nibs[16];    /* 16 bytes: 32 nibbles packed into 16 bytes */
} vqf_block_q4_0;        /* Total: 20 bytes */
```

Each byte packs two values: low nibble = even index, high nibble = odd index. Values are stored with a +8 offset (unsigned 0-15 range, centered at 8).

**Math:**
- Quantize: `scale = max(|values|) / 7`, `quant[i] = clamp(round(value[i] / scale) + 8, 0, 15)`
- Dequantize: `value[i] = scale * (nibble - 8)`

### Q4_K (256 values/super-block, 148 bytes)

The most sophisticated format, with two-level quantization: a super-block of 256 values divided into 8 sub-blocks of 32 values each.

```c
typedef struct {
    float   super_scale;   /* 4 bytes: scale for sub-block scales */
    float   super_min;     /* 4 bytes: scale for sub-block minimums */
    uint8_t scales[12];    /* 12 bytes: 8 pairs of 6-bit (scale, min) packed */
    uint8_t nibs[128];     /* 128 bytes: 256 nibbles */
} vqf_block_q4_k;          /* Total: 148 bytes */
```

**Packed 6-bit scales layout** (12 bytes encode 8 sub-block scale+min pairs):

```
For i in [0..4):
  byte[3*i + 0] = (s0 & 0x3F) | ((s1 & 0x03) << 6)
  byte[3*i + 1] = ((s1 >> 2) & 0x0F) | ((m0 & 0x0F) << 4)
  byte[3*i + 2] = ((m0 >> 4) & 0x03) | ((m1 & 0x3F) << 2)

where s0, s1 = 6-bit quantized scales for two sub-blocks
      m0, m1 = 6-bit quantized minimums for two sub-blocks
```

**Math:**
- Per sub-block: `sub_scale = (max - min) / 15`
- Super-block: `super_scale = max(all sub_scales) / 63`, `super_min = max(|all sub_mins|) / 63`
- 6-bit: `q_scale = round(sub_scale / super_scale)`, `q_min = round(|sub_min| / super_min)`
- 4-bit value: `quant = round((value - min) / effective_scale)`, clamped to [0, 15]
- Dequantize: `value = q_scale * super_scale * nibble - q_min * super_min`

---

## 3. Python Quantization Pipeline

**Script:** `quantize/quantize.py`

### Usage

```bash
python quantize.py <model_dir> <output.vqf> --type Q4_K
```

Where `model_dir` contains `consolidated.safetensors` and `tekken.json`.

### Which Tensors Get Quantized

Only **linear layer weights** in the encoder and decoder transformer layers are quantized:

```python
def should_quantize(name):
    if not name.endswith('.weight'):
        return False
    weight_kind = name.split('.')[-2]
    if weight_kind not in ('wq', 'wk', 'wv', 'wo', 'w1', 'w2', 'w3'):
        return False
    return 'layers' in name.split('.')
```

**Quantized:** Attention projections (wq, wk, wv, wo) and FFN weights (w1, w2, w3) in all 32 encoder and 26 decoder layers.

**NOT quantized (kept as BF16):** Token embeddings, biases, normalization weights, adapter weights, conv stem weights, adaptive norm weights.

### Pipeline Steps

1. **Parse safetensors header** (JSON metadata at start of file) to get tensor shapes and offsets without loading all data
2. **Compute VQF layout**: calculate quantized sizes, assign data offsets for each tensor
3. **Write VQF header + tensor descriptors** with 64-byte-aligned data section
4. **Stream each tensor**: load from safetensors one at a time, quantize on GPU (CUDA), write to VQF file, free memory

### GPU-Accelerated Quantization

All quantization functions run on CUDA when available. Example for Q8_0:

```python
def quantize_q8_0(t):
    values = t.flatten().to(device=device, dtype=torch.float32)
    # Pad to block boundary
    pad = (32 - values.numel() % 32) % 32
    if pad:
        values = torch.nn.functional.pad(values, (0, pad))

    blocks = values.reshape(-1, 32)
    amax = blocks.abs().amax(dim=1)
    scales = torch.where(amax > 0, amax / 127.0, amax.new_zeros(()))
    inv_scales = torch.where(scales > 0, 1.0 / scales, scales.new_zeros(()))
    quants = (blocks * inv_scales[:, None]).round().clamp(-127, 127).to(torch.int8)

    # Pack into 36-byte blocks: [4B scale | 32B quants]
    out = np.empty((n_blocks, 36), dtype=np.uint8)
    out[:, :4] = scales.cpu().numpy().view(np.uint8).reshape(-1, 4)
    out[:, 4:] = quants.cpu().numpy().view(np.uint8)
    return out.tobytes()
```

### BF16 Tensor Handling

Non-quantized tensors are stored as raw BF16 bytes:

```python
def tensor_raw_bytes(t):
    t = t.contiguous()
    if t.dtype == torch.float32:
        return t.numpy(force=True).tobytes()
    # BF16: reinterpret as int16 to get the raw bytes
    return t.view(torch.int16).numpy().tobytes()
```

### Output Size Comparison (Voxtral 4B)

| Format | Approximate Size |
|--------|-----------------|
| Original (BF16 safetensors) | ~8 GB |
| Q8_0 VQF | ~4.5 GB |
| Q4_0 VQF | ~2.5 GB |
| Q4_K VQF | ~2.3 GB |

---

## 4. Runtime Loading

**File:** `voxtral_quant_loader.c`

### Memory-Mapped Access

The VQF file is memory-mapped at load time for zero-copy weight access:

```c
vqf_mapped_file_t *vqf_open(const char *path) {
    // 1. Open file, get size
    // 2. Memory-map entire file (mmap on Linux, MapViewOfFile on Windows)
    // 3. Parse header: validate magic (VQF1) and version (1)
    // 4. Parse tensor descriptors from header section
    // Return handle with pointers into mmap'd region
}
```

### Weight Loading Modes

Three loading strategies depending on tensor type:

1. **Quantized tensors** (Q8_0/Q4_0/Q4_K): Return a direct pointer into the mmap'd region. No allocation, no copy.

2. **BF16 tensors**: Either return a direct `uint16_t*` pointer (for CUDA BF16 matmul), or convert to F32 on demand:
   ```c
   // BF16 -> F32: shift left 16 bits (zero-fill mantissa)
   uint32_t bits = ((uint32_t)bf16[i]) << 16;
   memcpy(&out[i], &bits, sizeof(float));
   ```

3. **F32 tensors**: Direct memcpy from mmap'd region.

### Layer Loading

Each encoder/decoder layer loads 7 quantized weight matrices plus F32 biases and norms:

```c
// Per-layer quantized weights (pointers into mmap)
l->wq_weight_q = load_quant(vf, "encoder.layers.0.attention.wq.weight",
                             &l->wq_qtype, &l->wq_numel);
// ... wk, wv, wo, w1, w2, w3

// Per-layer F32 tensors (allocated + converted from BF16)
l->wq_bias = load_f32(vf, "encoder.layers.0.attention.wq.bias");
l->attention_norm = load_f32(vf, "encoder.layers.0.attention_norm.weight");
```

---

## 5. CPU Dequantization Kernels

**File:** `voxtral_quant_kernels.c`

Runtime dequantization is fused with matrix-vector multiplication (GEMV). Weights are never fully dequantized into a separate buffer.

### Unified Dispatch

```c
void vox_linear_nobias_quant(float *y, const float *x, const void *W_q,
                              int seq_len, int in_dim, int out_dim, int qtype) {
    // Try CUDA first
    if (vox_cuda_quant_matmul_t(y, x, W_q, seq_len, in_dim, out_dim, qtype))
        return;

    // CPU fallback: per-token GEMV
    for (int s = 0; s < seq_len; s++) {
        switch (qtype) {
            case VQF_TYPE_Q8_0: matvec_q8_0(yi, xi, W_q, in_dim, out_dim); break;
            case VQF_TYPE_Q4_0: matvec_q4_0(yi, xi, W_q, in_dim, out_dim); break;
            case VQF_TYPE_Q4_K: matvec_q4_k(yi, xi, W_q, in_dim, out_dim); break;
        }
    }
}
```

### Q8_0 GEMV (innermost loop)

```c
for (int b = 0; b < blocks_per_row; b++) {
    float scale = *(float*)block;
    int8_t *quants = block + 4;
    float partial = 0.0f;
    for (int i = 0; i < 32; i++)
        partial += x[k_base + i] * (float)quants[i];
    acc += scale * partial;
}
```

### Q4_0 GEMV

```c
for (int b = 0; b < blocks_per_row; b++) {
    float scale = *(float*)block;
    uint8_t *nibs = block + 4;
    float partial = 0.0f;
    for (int i = 0; i < 16; i++) {
        int lo = (nibs[i] & 0xF) - 8;
        int hi = ((nibs[i] >> 4) & 0xF) - 8;
        partial += x[k_base + 2*i] * (float)lo;
        partial += x[k_base + 2*i + 1] * (float)hi;
    }
    acc += scale * partial;
}
```

### Q4_K GEMV

```c
for (int sb = 0; sb < sblocks_per_row; sb++) {
    float super_scale = *(float*)block;
    float super_min   = *(float*)(block + 4);
    // Unpack 6-bit scales and mins from 12-byte packed section
    uint8_t q_scales[8], q_mins[8];
    // ... (6-bit unpacking, see block structure above)

    for (int sub = 0; sub < 8; sub++) {
        float s = (float)q_scales[sub] * super_scale;
        float m = (float)q_mins[sub] * super_min;
        for (int i = 0; i < 16; i++) {
            int lo = nibs[i] & 0xF;
            int hi = (nibs[i] >> 4) & 0xF;
            partial += x[sk + 2*i]     * (s * lo - m);
            partial += x[sk + 2*i + 1] * (s * hi - m);
        }
    }
}
```

---

## 6. CUDA Dequantization

**File:** `voxtral_cuda_quant.c`

### Permanent GPU Weight Upload

All quantized weights are uploaded to GPU VRAM at model load time and kept there permanently. This avoids per-inference H2D transfers:

```c
int vox_cuda_quant_upload_all(vox_ctx_t *ctx) {
    // Upload all encoder layer weights (32 layers x 7 matrices)
    for (int i = 0; i < VOX_ENC_LAYERS; i++) {
        UPLOAD(wq); UPLOAD(wk); UPLOAD(wv); UPLOAD(wo);
        UPLOAD(w1); UPLOAD(w2); UPLOAD(w3);
    }
    // Upload all decoder layer weights (26 layers x 7 matrices)
    for (int i = 0; i < VOX_DEC_LAYERS; i++) { ... }
}
```

A host-pointer-to-device-pointer cache avoids duplicate uploads.

### CUDA GEMV Kernels

Custom PTX kernels for each quantization type, loaded from embedded cubin:

```c
cuModuleGetFunction(&g_fn_gemv_q8_0, g_mod, "vox_gemv_q8_0");
cuModuleGetFunction(&g_fn_gemv_q4_0, g_mod, "vox_gemv_q4_0");
cuModuleGetFunction(&g_fn_gemv_q4_k, g_mod, "vox_gemv_q4_k");
// Plus _beta variants for fused bias addition
```

### GPU GEMV Dispatch

```c
int vox_cuda_quant_matmul_t(float *out, const float *x, const void *W_q,
                             int M, int K, int N, int qtype) {
    CUdeviceptr dW = vox_cuda_quant_weight_get(W_q);  // Lookup cached device ptr
    if (!dW) return 0;

    // Upload activation x to GPU
    cuMemcpyHtoDAsync(g_qx_dev, x, M * K * sizeof(float), g_stream);

    // Launch GEMV kernel per sequence token
    for (int s = 0; s < M; s++)
        vox_cuda_quant_gemv_dev(dOs, dXs, dW, K, N, qtype);

    // Download result
    cuMemcpyDtoHAsync(out, g_qout_dev, M * N * sizeof(float), g_stream);
    cuStreamSynchronize(g_stream);
    return 1;
}
```

Each GEMV kernel processes 16-32 output rows per thread block with 256 threads, using padded shared memory (`(K + K/32) * sizeof(float)`) for the input activation vector. The padding eliminates 32-way bank conflicts by shifting each 32-element row by one float (`SH_PAD(idx) = idx + idx/32`). Weight data is loaded using vectorized `uint4` reads (16 bytes per load) and unpacked in registers via bit shifts, reducing memory transactions by 8-16x compared to individual byte loads.

---

## 7. Inference-Time Weight Selection

At each linear layer, the code checks whether quantized or BF16 weights are available:

```c
// Encoder linear (example)
if (l->wq_weight_q)
    vox_linear_nobias_quant(out, in, l->wq_weight_q, seq, in_d, out_d, l->wq_qtype);
else
    vox_linear_nobias_bf16(out, in, l->wq_weight_bf16, seq, in_d, out_d);
```

This allows mixed-precision operation: quantized linear layers with BF16 embeddings, norms, and adapter weights.
