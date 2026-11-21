// ternary-gemm: fused ternary (BitNet b1.58-style) matmul kernels for LLM decode.
//
// Compiled at runtime with NVRTC (see src/gpu.rs). Self-contained: no #include at all, so NVRTC
// needs no CUDA headers on disk. fp16 values are carried as raw 16-bit patterns (`half_t`) and
// converted with PTX cvt instructions.
//
// ---------------------------------------------------------------------------------
// Weight format ("permuted" layout, used by the fast kernels)
//   * W is [N out][K in], every weight w in {-1, 0, +1}, stored as code c = w + 1 in {0,1,2}.
//   * 16 codes per u32. Inside a word, weight (16*kb + 4*j + i) lives at bits [8*i + 2*j, +2).
//     So (word >> 2*j) & 0x03030303 yields 4 bytes = codes of 4 *consecutive* weights,
//     ready for __dp4a against 4 consecutive int8 activations.
//   * Row = K/16 words, split into chunks of 128 words (2048 weights; the last chunk may be
//     shorter, always a multiple of 4 words). In a chunk of Wc words, with L = Wc/4 lanes,
//     the word for k-block (q*L + l) is stored at chunk offset 4*l + q. Lane l therefore
//     reads ONE uint4 (4 words, 16B) per row per chunk, and for each q the active lanes read
//     a contiguous, conflict-free run of activations.
//   * Because codes are w+1, sum_k c_k*x_k = sum_k w_k*x_k + sum_k x_k. The kernels subtract
//     the per-token activation sum (computed once, during activation quantization).
//
// Activation format (int8 path): BitNet per-token absmax.
//   s = 127 / max(absmax(x), 1e-5);  xq = clamp(rint(x * s), -128, 127);  y = acc / s * w_scale.
// Optional RMSNorm is fused in front: v = (x * gamma) * rsqrt(mean(x^2) + eps).
// ---------------------------------------------------------------------------------

typedef unsigned short half_t;  // IEEE fp16 bit pattern

__device__ __forceinline__ float h2f(half_t h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
__device__ __forceinline__ half_t f2h_rn(float f) {
    half_t h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}
// Two packed fp16 values (low half first) -> float2.
__device__ __forceinline__ float2 h22f2(unsigned v) {
    return make_float2(h2f((half_t)(v & 0xffffu)), h2f((half_t)(v >> 16)));
}

#define CHUNK_WORDS 128

__device__ __forceinline__ int warp_sum_i(int v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_sum_f(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max_f(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Block-wide reductions. `scratch` must hold >= 32 elements. Every thread gets the result.
__device__ __forceinline__ float block_sum_f(float v, float* scratch) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warp_sum_f(v);
    __syncthreads();
    if (lane == 0) scratch[wid] = v;
    __syncthreads();
    if (wid == 0) {
        float r = lane < nw ? scratch[lane] : 0.f;
        r = warp_sum_f(r);
        if (lane == 0) scratch[0] = r;
    }
    __syncthreads();
    return scratch[0];
}
__device__ __forceinline__ float block_max_f(float v, float* scratch) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warp_max_f(v);
    __syncthreads();
    if (lane == 0) scratch[wid] = v;
    __syncthreads();
    if (wid == 0) {
        float r = lane < nw ? scratch[lane] : 0.f;
        r = warp_max_f(r);
        if (lane == 0) scratch[0] = r;
    }
    __syncthreads();
    return scratch[0];
}
__device__ __forceinline__ int block_sum_i(int v, int* scratch) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warp_sum_i(v);
    __syncthreads();
    if (lane == 0) scratch[wid] = v;
    __syncthreads();
    if (wid == 0) {
        int r = lane < nw ? scratch[lane] : 0;
        r = warp_sum_i(r);
        if (lane == 0) scratch[0] = r;
    }
    __syncthreads();
    return scratch[0];
}

// Quantize one token (K values) with the whole block. Optional fused RMSNorm (gamma != null).
// Writes K int8 values to dst and returns (s, sum) through the out-params (valid in all threads).
// K must be a multiple of 64 (so 2 x fp16 / char2 accesses are aligned).
__device__ __forceinline__ void quantize_token(const half_t* __restrict__ x, const half_t* __restrict__ gamma,
                                               float eps, int K, signed char* dst, float* out_s, int* out_sum,
                                               float* fscratch, int* iscratch) {
    const unsigned* x2 = reinterpret_cast<const unsigned*>(x);
    const unsigned* g2 = reinterpret_cast<const unsigned*>(gamma);
    const int K2 = K >> 1;
    float ss = 0.f, m = 0.f;
    for (int i = threadIdx.x; i < K2; i += blockDim.x) {
        float2 v = h22f2(x2[i]);
        if (gamma) {
            float2 g = h22f2(g2[i]);
            ss = fmaf(v.x, v.x, ss);
            ss = fmaf(v.y, v.y, ss);
            m = fmaxf(m, fabsf(__fmul_rn(v.x, g.x)));
            m = fmaxf(m, fabsf(__fmul_rn(v.y, g.y)));
        } else {
            m = fmaxf(m, fmaxf(fabsf(v.x), fabsf(v.y)));
        }
    }
    float inv = 1.f;
    if (gamma) {
        ss = block_sum_f(ss, fscratch);
        inv = rsqrtf(ss / (float)K + eps);
    }
    m = block_max_f(m, fscratch);
    // max_k |fl(a_k * inv)| == fl(max_k |a_k| * inv) because rounding is monotone.
    const float absmax = fmaxf(gamma ? __fmul_rn(m, inv) : m, 1e-5f);
    const float s = __fdiv_rn(127.f, absmax);
    int sum = 0;
    char2* d2 = reinterpret_cast<char2*>(dst);
    for (int i = threadIdx.x; i < K2; i += blockDim.x) {
        float2 v = h22f2(x2[i]);
        if (gamma) {
            float2 g = h22f2(g2[i]);
            v.x = __fmul_rn(__fmul_rn(v.x, g.x), inv);
            v.y = __fmul_rn(__fmul_rn(v.y, g.y), inv);
        }
        int q0 = min(127, max(-128, __float2int_rn(__fmul_rn(v.x, s))));
        int q1 = min(127, max(-128, __float2int_rn(__fmul_rn(v.y, s))));
        sum += q0 + q1;
        d2[i] = make_char2((signed char)q0, (signed char)q1);
    }
    sum = block_sum_i(sum, iscratch);
    *out_s = s;
    *out_sum = sum;
}

__device__ __forceinline__ void store_out(half_t* __restrict__ out, const half_t* __restrict__ residual, size_t idx, float v) {
    if (residual) v = __fadd_rn(v, h2f(residual[idx]));
    out[idx] = f2h_rn(v);
}

__device__ __forceinline__ unsigned word_of(const uint4& v, int q) {
    return q == 0 ? v.x : (q == 1 ? v.y : (q == 2 ? v.z : v.w));
}

// ---------------------------------------------------------------------------------
// int8-activation main loop. One warp computes R consecutive output rows for up to BT tokens.
// xq / xs / xsum may point to global or shared memory (generic addressing).
// ---------------------------------------------------------------------------------
template <int BT, int R>
__device__ __forceinline__ void ternary_i8_warp(const unsigned* __restrict__ w, const float* __restrict__ wscale,
                                                const signed char* xq, const float* xs, const int* xsum,
                                                const half_t* __restrict__ residual, half_t* __restrict__ out,
                                                int N, int K, int batch, int row0) {
    const int lane = threadIdx.x & 31;
    const int KW = K >> 4;
    int acc[R][BT];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int b = 0; b < BT; ++b) acc[r][b] = 0;

    for (int base = 0; base < KW; base += CHUNK_WORDS) {
        const int Wc = min(CHUNK_WORDS, KW - base);
        const int L = Wc >> 2;
        if (lane < L) {
            int codes[R][16];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int row = row0 + r;
                uint4 wv = make_uint4(0u, 0u, 0u, 0u);
                if (row < N) wv = __ldg(reinterpret_cast<const uint4*>(w + (size_t)row * KW + base) + lane);
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const unsigned word = word_of(wv, q);
#pragma unroll
                    for (int j = 0; j < 4; ++j) codes[r][4 * q + j] = (int)((word >> (2 * j)) & 0x03030303u);
                }
            }
#pragma unroll
            for (int b = 0; b < BT; ++b) {
                if (b < batch) {
                    const int4* xb = reinterpret_cast<const int4*>(xq + (size_t)b * K);
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int4 xv = xb[base + q * L + lane];
#pragma unroll
                        for (int r = 0; r < R; ++r) {
                            int a = acc[r][b];
                            a = __dp4a(codes[r][4 * q + 0], xv.x, a);
                            a = __dp4a(codes[r][4 * q + 1], xv.y, a);
                            a = __dp4a(codes[r][4 * q + 2], xv.z, a);
                            a = __dp4a(codes[r][4 * q + 3], xv.w, a);
                            acc[r][b] = a;
                        }
                    }
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int b = 0; b < BT; ++b) acc[r][b] = warp_sum_i(acc[r][b]);

    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row0 + r;
            if (row >= N) continue;
            const float ws = wscale[row];
#pragma unroll
            for (int b = 0; b < BT; ++b) {
                if (b < batch) {
                    float t = (float)(acc[r][b] - xsum[b]);
                    t = __fdiv_rn(t, xs[b]);
                    t = __fmul_rn(t, ws);
                    store_out(out, residual, (size_t)b * N + row, t);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------
// fp16-activation main loop (W2A16). Weights decoded with a magic-number trick: a code byte c
// spliced under 0x4B000000 is the float 2^23 + c, and subtracting 2^23 + 1 gives w = c - 1 exactly
// (one PRMT + one FADD per weight, no int->float conversion).
// ---------------------------------------------------------------------------------
template <int BT, int R>
__device__ __forceinline__ void ternary_f16_warp(const unsigned* __restrict__ w, const float* __restrict__ wscale,
                                                 const half_t* __restrict__ x, const half_t* __restrict__ residual,
                                                 half_t* __restrict__ out, int N, int K, int batch, int row0) {
    const int lane = threadIdx.x & 31;
    const int KW = K >> 4;
    float acc[R][BT];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int b = 0; b < BT; ++b) acc[r][b] = 0.f;

    for (int base = 0; base < KW; base += CHUNK_WORDS) {
        const int Wc = min(CHUNK_WORDS, KW - base);
        const int L = Wc >> 2;
        if (lane < L) {
            uint4 wv[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int row = row0 + r;
                wv[r] = make_uint4(0x55555555u, 0x55555555u, 0x55555555u, 0x55555555u);  // code 1 == weight 0
                if (row < N) wv[r] = __ldg(reinterpret_cast<const uint4*>(w + (size_t)row * KW + base) + lane);
            }
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int kb = base + q * L + lane;  // 16-weight block index
                float wf[R][16];
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const unsigned word = word_of(wv[r], q);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const unsigned c = (word >> (2 * j)) & 0x03030303u;
#pragma unroll
                        for (int i = 0; i < 4; ++i)  // 0x4B000000 | c  ==  2^23 + c  as float
                            wf[r][4 * j + i] = __int_as_float(__byte_perm(c, 0x4B000000u, 0x7650u | i)) - 8388609.f;
                    }
                }
#pragma unroll
                for (int b = 0; b < BT; ++b) {
                    if (b < batch) {
                        const int4* xp = reinterpret_cast<const int4*>(x + (size_t)b * K + 16 * (size_t)kb);
                        const int4 xa = __ldg(xp), xc = __ldg(xp + 1);
                        const unsigned xw[8] = {(unsigned)xa.x, (unsigned)xa.y, (unsigned)xa.z, (unsigned)xa.w,
                                                (unsigned)xc.x, (unsigned)xc.y, (unsigned)xc.z, (unsigned)xc.w};
#pragma unroll
                        for (int h = 0; h < 8; ++h) {
                            const float2 xf = h22f2(xw[h]);
#pragma unroll
                            for (int r = 0; r < R; ++r) {
                                acc[r][b] = fmaf(wf[r][2 * h], xf.x, acc[r][b]);
                                acc[r][b] = fmaf(wf[r][2 * h + 1], xf.y, acc[r][b]);
                            }
                        }
                    }
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int b = 0; b < BT; ++b) acc[r][b] = warp_sum_f(acc[r][b]);

    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row0 + r;
            if (row >= N) continue;
            const float ws = wscale[row];
#pragma unroll
            for (int b = 0; b < BT; ++b)
                if (b < batch) store_out(out, residual, (size_t)b * N + row, __fmul_rn(acc[r][b], ws));
        }
    }
}

// ---------------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------------

// Activation quantization (+ optional RMSNorm). grid = batch, any block size that is a multiple of 32.
extern "C" __global__ void __launch_bounds__(1024) quantize_i8(const half_t* __restrict__ x, const half_t* __restrict__ gamma,
                                                                float eps, int K, signed char* __restrict__ xq,
                                                                float* __restrict__ xs, int* __restrict__ xsum) {
    __shared__ float fscratch[32];
    __shared__ int iscratch[32];
    const int b = blockIdx.x;
    float s;
    int sum;
    quantize_token(x + (size_t)b * K, gamma, eps, K, xq + (size_t)b * K, &s, &sum, fscratch, iscratch);
    if (threadIdx.x == 0) {
        xs[b] = s;
        xsum[b] = sum;
    }
}

#define DEFINE_I8(BT, R)                                                                                          \
    extern "C" __global__ void __launch_bounds__(256)                                                             \
        ternary_i8_b##BT##_r##R(const unsigned* __restrict__ w, const float* __restrict__ wscale,                 \
                                const signed char* __restrict__ xq, const float* __restrict__ xs,                 \
                                const int* __restrict__ xsum, const half_t* __restrict__ residual,                  \
                                half_t* __restrict__ out, int N, int K, int batch) {                                \
        const int row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * R;                              \
        if (row0 >= N) return;                                                                                    \
        ternary_i8_warp<BT, R>(w, wscale, xq, xs, xsum, residual, out, N, K, batch, row0);                        \
    }                                                                                                             \
    /* Single launch: every block re-quantizes the (tiny) activations into shared memory first. */                \
    extern "C" __global__ void __launch_bounds__(256)                                                             \
        ternary_i8_fused_b##BT##_r##R(const unsigned* __restrict__ w, const float* __restrict__ wscale,           \
                                      const half_t* __restrict__ x, const half_t* __restrict__ gamma, float eps,      \
                                      const half_t* __restrict__ residual, half_t* __restrict__ out, int N, int K,    \
                                      int batch) {                                                                \
        extern __shared__ int4 smem_i4[];                                                                         \
        signed char* xq_s = reinterpret_cast<signed char*>(smem_i4);                                              \
        __shared__ float s_scale[BT];                                                                             \
        __shared__ int s_sum[BT];                                                                                 \
        __shared__ float fscratch[32];                                                                            \
        __shared__ int iscratch[32];                                                                              \
        for (int b = 0; b < batch; ++b) {                                                                         \
            float s;                                                                                              \
            int sum;                                                                                              \
            quantize_token(x + (size_t)b * K, gamma, eps, K, xq_s + (size_t)b * K, &s, &sum, fscratch, iscratch); \
            if (threadIdx.x == 0) {                                                                               \
                s_scale[b] = s;                                                                                   \
                s_sum[b] = sum;                                                                                   \
            }                                                                                                     \
        }                                                                                                         \
        __syncthreads();                                                                                          \
        const int row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * R;                              \
        if (row0 >= N) return;                                                                                    \
        ternary_i8_warp<BT, R>(w, wscale, xq_s, s_scale, s_sum, residual, out, N, K, batch, row0);                \
    }                                                                                                             \
    extern "C" __global__ void __launch_bounds__(256)                                                             \
        ternary_f16_b##BT##_r##R(const unsigned* __restrict__ w, const float* __restrict__ wscale,                \
                                 const half_t* __restrict__ x, const half_t* __restrict__ residual,                   \
                                 half_t* __restrict__ out, int N, int K, int batch) {                               \
        const int row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * R;                              \
        if (row0 >= N) return;                                                                                    \
        ternary_f16_warp<BT, R>(w, wscale, x, residual, out, N, K, batch, row0);                                  \
    }

#define DEFINE_ALL_R(BT) DEFINE_I8(BT, 1) DEFINE_I8(BT, 2) DEFINE_I8(BT, 4)
DEFINE_ALL_R(1)
DEFINE_ALL_R(2)
DEFINE_ALL_R(4)
DEFINE_ALL_R(8)
DEFINE_ALL_R(16)

// ---------------------------------------------------------------------------------
// Naive baselines: one thread per (token, row), "natural" 2-bit layout
// (weight k at word k/16, bits 2*(k%16)), scalar decode + multiply. Same math, same output bits.
// ---------------------------------------------------------------------------------
extern "C" __global__ void naive_i8(const unsigned* __restrict__ w_nat, const float* __restrict__ wscale,
                                    const signed char* __restrict__ xq, const float* __restrict__ xs,
                                    const half_t* __restrict__ residual, half_t* __restrict__ out, int N, int K,
                                    int batch) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * batch) return;
    const int row = idx % N, b = idx / N;
    const unsigned* wr = w_nat + (size_t)row * (K >> 4);
    const signed char* xb = xq + (size_t)b * K;
    int acc = 0;
    for (int k = 0; k < K; ++k) {
        const int c = (int)((wr[k >> 4] >> (2 * (k & 15))) & 3u);
        acc += (c - 1) * (int)xb[k];
    }
    float t = (float)acc;
    t = __fdiv_rn(t, xs[b]);
    t = __fmul_rn(t, wscale[row]);
    store_out(out, residual, (size_t)b * N + row, t);
}

extern "C" __global__ void naive_f16(const unsigned* __restrict__ w_nat, const float* __restrict__ wscale,
                                     const half_t* __restrict__ x, const half_t* __restrict__ residual,
                                     half_t* __restrict__ out, int N, int K, int batch) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * batch) return;
    const int row = idx % N, b = idx / N;
    const unsigned* wr = w_nat + (size_t)row * (K >> 4);
    const half_t* xb = x + (size_t)b * K;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        const int c = (int)((wr[k >> 4] >> (2 * (k & 15))) & 3u);
        acc = fmaf((float)(c - 1), h2f(xb[k]), acc);
    }
    store_out(out, residual, (size_t)b * N + row, __fmul_rn(acc, wscale[row]));
}

// ---------------------------------------------------------------------------------
// Benchmark helpers: fill buffers on-device with pseudo-random data (avoids benchmarking
// all-zero buffers, which some GPUs can compress).
// ---------------------------------------------------------------------------------
__device__ __forceinline__ unsigned hash_u32(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}
extern "C" __global__ void fill_ternary_codes(unsigned* p, unsigned long long n, unsigned seed) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned h = hash_u32((unsigned)i * 0x9E3779B9u + seed);
        p[i] = h & ~((h >> 1) & 0x55555555u);  // code 3 -> 2, so every 2-bit field is in {0,1,2}
    }
}
extern "C" __global__ void fill_f16(half_t* p, unsigned long long n, unsigned seed) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned h = hash_u32((unsigned)i * 0x9E3779B9u + seed);
        p[i] = f2h_rn(((float)(h & 0xffffu) / 65536.f - 0.5f) * 0.05f);
    }
}
extern "C" __global__ void fill_f32(float* p, unsigned long long n, unsigned seed) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned h = hash_u32((unsigned)i * 0x9E3779B9u + seed);
        p[i] = 0.5f + (float)(h & 0xffffu) / 65536.f;
    }
}

// =================================================================================
// Latency-first small-model path (quant / signal models): a whole ternary MLP in ONE block.
//
// Weight layout ("k-major", per layer): word w (k-block of 16 weights) of row n is at
// W[layer_off + w * N + n], same in-word interleave as above. Thread t owns rows t, t+T, ...,
// so a warp reads 32 consecutive words (coalesced) and every lane reads the same activation
// word (shared-memory broadcast). Layers are chained through shared memory, so a K, N <= 4096
// model of any depth is one launch with no global round trips between layers.
//
// Per layer (bit-exact against src/mlp.rs):
//   s = 127 / max(absmax(x), 1e-5);  q = clamp(rint(x * s), -128, 127)
//   y = ((acc - sum_q) / s) * scale[n] + bias[n];  act = identity | relu | relu^2
// desc[4*l .. 4*l+3] = {N, K, act, unused}; sb = per layer [scale[N], bias[N]] back to back.
// =================================================================================

#define MLP_MAX_DIM 4096
#define ACT_IDENTITY 0
#define ACT_RELU 1
#define ACT_RELU2 2

__device__ __forceinline__ float mlp_act(float v, int act) {
    if (act == ACT_RELU) return v > 0.0f ? v : 0.0f;
    if (act == ACT_RELU2) { float r = v > 0.0f ? v : 0.0f; return __fmul_rn(r, r); }
    return v;
}

// Block-wide reductions (blockDim.x a multiple of 32, <= 1024). `red` holds >= 32 entries.
__device__ __forceinline__ float mlp_block_max(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    v = l < nw ? red[l] : 0.0f;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ int mlp_block_sum(int v, int* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    v = l < nw ? red[l] : 0;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// One forward pass for one sample. x: [in_dim] (may be mapped host memory, read volatile),
// y: [out_dim]. smem: 2 * max_dim floats followed by max_dim bytes of int8 activations.
__device__ void mlp_forward(const int* __restrict__ desc, int layers, const unsigned* __restrict__ W,
                            const float* __restrict__ sb, const volatile float* x, int in_dim, float* y,
                            int max_dim, unsigned char* smem) {
    __shared__ float redf[32];
    __shared__ int redi[32];
    float* cur = reinterpret_cast<float*>(smem);
    float* nxt = cur + max_dim;
    int* qw = reinterpret_cast<int*>(nxt + max_dim);
    const int tid = threadIdx.x, T = blockDim.x;
    for (int i = tid; i < in_dim; i += T) cur[i] = x[i];
    __syncthreads();
    size_t woff = 0, soff = 0;
    for (int l = 0; l < layers; ++l) {
        const int n = desc[4 * l], k = desc[4 * l + 1], act = desc[4 * l + 2];
        float m = 0.0f;
        for (int i = tid; i < k; i += T) m = fmaxf(m, fabsf(cur[i]));
        m = mlp_block_max(m, redf);
        const float s = __fdiv_rn(127.0f, fmaxf(m, 1e-5f));
        int part = 0;
        for (int i = tid; i < (k >> 2); i += T) {
            unsigned packed = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int q = (int)rintf(__fmul_rn(cur[4 * i + j], s));
                q = q < -128 ? -128 : (q > 127 ? 127 : q);
                part += q;
                packed |= ((unsigned)q & 0xffu) << (8 * j);
            }
            qw[i] = (int)packed;
        }
        const int sum_q = mlp_block_sum(part, redi);  // also a barrier: qw is complete
        const unsigned* Wl = W + woff;
        const int kw = k >> 4;
        const bool last = l == layers - 1;
        for (int r = tid; r < n; r += T) {
            int acc = 0;
#pragma unroll 4
            for (int w = 0; w < kw; ++w) {
                const unsigned word = __ldg(Wl + (size_t)w * n + r);
                acc = __dp4a((int)(word & 0x03030303u), qw[4 * w + 0], acc);
                acc = __dp4a((int)((word >> 2) & 0x03030303u), qw[4 * w + 1], acc);
                acc = __dp4a((int)((word >> 4) & 0x03030303u), qw[4 * w + 2], acc);
                acc = __dp4a((int)((word >> 6) & 0x03030303u), qw[4 * w + 3], acc);
            }
            float v = __fdiv_rn((float)(acc - sum_q), s);
            v = __fmul_rn(v, __ldg(sb + soff + r));
            v = __fadd_rn(v, __ldg(sb + soff + n + r));
            v = mlp_act(v, act);
            if (last) y[r] = v; else nxt[r] = v;
        }
        woff += (size_t)n * kw;
        soff += 2 * (size_t)n;
        float* t = cur; cur = nxt; nxt = t;
        __syncthreads();
    }
}

// One block per sample: x [batch][in_dim] -> y [batch][out_dim].
extern "C" __global__ void __launch_bounds__(1024) ternary_mlp(const int* __restrict__ desc, int layers,
        const unsigned* __restrict__ W, const float* __restrict__ sb, const float* x, float* y,
        int in_dim, int out_dim, int max_dim) {
    extern __shared__ __align__(16) unsigned char mlp_smem[];
    mlp_forward(desc, layers, W, sb, x + (size_t)blockIdx.x * in_dim, in_dim,
                y + (size_t)blockIdx.x * out_dim, max_dim, mlp_smem);
}

// EXPERIMENTAL persistent mode: one block stays resident and spins on a doorbell in mapped
// (zero-copy) pinned host memory, so a request costs no kernel launch and no memcpy calls.
// ctrl[0] = request sequence number written by the host (0xffffffff = stop),
// ctrl[1] = last completed sequence number written by the device.
// x / y are mapped host buffers (the host writes x before ringing, reads y after done).
extern "C" __global__ void __launch_bounds__(1024) ternary_mlp_persistent(const int* __restrict__ desc,
        int layers, const unsigned* __restrict__ W, const float* __restrict__ sb, volatile unsigned* ctrl,
        const float* x, float* y, int in_dim, int out_dim, int max_dim) {
    extern __shared__ __align__(16) unsigned char mlp_smem[];
    __shared__ unsigned seq_s;
    unsigned last = ctrl[1];
    for (;;) {
        if (threadIdx.x == 0) {
            unsigned v;
            do { v = ctrl[0]; } while (v == last);
            __threadfence_system();
            seq_s = v;
        }
        __syncthreads();
        const unsigned v = seq_s;
        if (v == 0xffffffffu) return;
        mlp_forward(desc, layers, W, sb, (const volatile float*)x, in_dim, y, max_dim, mlp_smem);
        __threadfence_system();  // y visible to the host before the completion flag
        __syncthreads();
        if (threadIdx.x == 0) ctrl[1] = v;
        last = v;
        __syncthreads();
    }
}

// Helpers for the cuBLAS FP16 MLP baseline.
extern "C" __global__ void f32_to_f16(const float* __restrict__ x, half_t* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = f2h_rn(x[i]);
}
extern "C" __global__ void bias_act_f16(half_t* __restrict__ y, const float* __restrict__ bias, int n, int act,
                                        float* __restrict__ out_f32) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = mlp_act(h2f(y[i]) + bias[i], act);
        if (out_f32) out_f32[i] = v; else y[i] = f2h_rn(v);
    }
}
