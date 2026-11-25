// ternary-gemm: extra GEMV kernels for batch-1 decode (sm_80+ unless noted).
//
// Self-contained (no #include) so NVRTC compiles it without CUDA headers; the Rust side compiles
// it with gpu::compile_ext. Weight formats are the ones in kernels/ternary.cu:
//   * permuted: 16 codes (w + 1) per u32, weight 16*kb + 4*j + i at bits [8*i + 2*j, +2), rows in
//     128-word chunks where lane l of a warp owns the uint4 at chunk offset 4*l (k-blocks
//     q*L + l, L = Wc / 4).
//   * natural:  weight k at word k / 16, bits [2*(k % 16), +2).
// Activations come from quantize_i8 (int8 per-token absmax, s, sum). Every kernel accumulates
// sum_k code_k * q_k in int32 (exact, order independent), then runs the same epilogue:
//   t = (acc - sum) / s * scale[row]       (IEEE rn div and mul, as in ternary.cu)
//   t = act(t)                             (0 identity, 1 relu, 2 relu^2: fused dequant + act)
//   t = t + residual[row]                  (optional, rn)
//   out[row] = fp16_rn(t)
// so all of them are bit-identical to cuda_ext::reference_gemv on the CPU.
//
// Kernels:
//   tgx_gemv_cpasync_r{1,2,4}  4-stage cp.async pipeline per warp, activations staged in smem.
//   tgx_gemv_splitk_r2 +       split-K over chunk ranges with int32 atomics, then a finalize pass
//   tgx_splitk_finalize        (for short-N / long-K shapes that cannot fill the SMs by rows).
//   tgx_gemv_lut               T-MAC style table lookup (2 weights -> one 16-entry table index).
//   tgx_gemv_warpspec          1 producer warp (cp.async) + 4 consumer warps (dp4a), named-barrier
//                              ring; the pre-Hopper warp-specialization pattern.

typedef unsigned short half_t;

#define TGX_CHUNK_WORDS 128

extern __shared__ int4 tgx_smem[];  // dynamic shared memory, shared by the kernels that use it

__device__ __forceinline__ float tgx_h2f(half_t h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
__device__ __forceinline__ half_t tgx_f2h_rn(float f) {
    half_t h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}
__device__ __forceinline__ int tgx_warp_sum(int v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ unsigned tgx_word(const uint4& v, int q) {
    return q == 0 ? v.x : q == 1 ? v.y : q == 2 ? v.z : v.w;
}

// Fused dequant + activation + residual epilogue (bit-exact contract above).
__device__ __forceinline__ void tgx_epilogue(int acc, int xsum, float xs, float ws, int act,
                                             const half_t* __restrict__ residual, half_t* __restrict__ out,
                                             int idx) {
    float t = (float)(acc - xsum);
    t = __fdiv_rn(t, xs);
    t = __fmul_rn(t, ws);
    if (act == 1) {
        t = t > 0.f ? t : 0.f;
    } else if (act == 2) {
        const float r = t > 0.f ? t : 0.f;
        t = __fmul_rn(r, r);
    }
    if (residual) t = __fadd_rn(t, tgx_h2f(residual[idx]));
    out[idx] = tgx_f2h_rn(t);
}

// ---- cp.async / named barrier primitives (inline PTX; sm_80+) ----
__device__ __forceinline__ unsigned tgx_smem_addr(const void* p) {
    unsigned long long a;
    asm("cvta.to.shared.u64 %0, %1;" : "=l"(a) : "l"(p));
    return (unsigned)a;
}
__device__ __forceinline__ void tgx_cp_async16(void* smem, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(tgx_smem_addr(smem)), "l"(gmem));
}
__device__ __forceinline__ void tgx_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void tgx_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
// Barrier ids are immediates so ptxas reserves only the ones used (a register id reserves all 16).
template <int ID, int THREADS>
__device__ __forceinline__ void tgx_bar_sync() {
    asm volatile("bar.sync %0, %1;\n" ::"n"(ID), "n"(THREADS) : "memory");
}
template <int ID, int THREADS>
__device__ __forceinline__ void tgx_bar_arrive() {
    asm volatile("bar.arrive %0, %1;\n" ::"n"(ID), "n"(THREADS) : "memory");
}

// Stage the int8 activations (K bytes, K % 16 == 0) into shared memory with cp.async; the caller
// commits the group.
__device__ __forceinline__ void tgx_stage_acts(signed char* xs_smem, const signed char* __restrict__ xq, int K) {
    for (int i = threadIdx.x; i < (K >> 4); i += blockDim.x) tgx_cp_async16(xs_smem + 16 * i, xq + 16 * i);
}

// dp4a over one chunk for one row: lane's uint4 of codes against the 4 activation int4s it needs.
__device__ __forceinline__ int tgx_chunk_dot(const uint4 wv, const signed char* xsm, int base, int L, int lane,
                                             int acc) {
    const int4* xb = reinterpret_cast<const int4*>(xsm);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const unsigned word = tgx_word(wv, q);
        const int4 xv = xb[base + q * L + lane];
        acc = __dp4a((int)((word >> 0) & 0x03030303u), xv.x, acc);
        acc = __dp4a((int)((word >> 2) & 0x03030303u), xv.y, acc);
        acc = __dp4a((int)((word >> 4) & 0x03030303u), xv.z, acc);
        acc = __dp4a((int)((word >> 6) & 0x03030303u), xv.w, acc);
    }
    return acc;
}

// =================================================================================================
// 1. cp.async pipelined GEMV. Block = TGX_CP_WARPS warps, warp w owns rows row0 .. row0 + R - 1.
//    Each lane copies its own uint4 per row per chunk into a STAGES-deep ring, so a lane only reads
//    smem it wrote itself: cp.async.wait_group is the only synchronization the ring needs.
//    Dynamic smem: K bytes of activations + TGX_CP_WARPS * STAGES * R * 512 bytes of weights.
// =================================================================================================
#define TGX_CP_WARPS 4
#define TGX_CP_STAGES 4

// Issue chunk c of this warp's R rows into ring slot c % STAGES (lane copies its own uint4), then
// commit a group (an empty one past the last chunk, so the wait_group arithmetic stays uniform).
template <int R>
__device__ __forceinline__ void tgx_cp_issue(const unsigned* __restrict__ w, uint4* ring, int row0, int N, int KW,
                                             int chunks, int c, int lane) {
    if (c < chunks) {
        const int base = c * TGX_CHUNK_WORDS;
        const int L = min(TGX_CHUNK_WORDS, KW - base) >> 2;
        if (lane < L) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int row = min(row0 + r, N - 1);  // clamp: rows past N are computed, not stored
                tgx_cp_async16(&ring[((c % TGX_CP_STAGES) * R + r) * 32 + lane],
                               reinterpret_cast<const uint4*>(w + (size_t)row * KW + base) + lane);
            }
        }
    }
    tgx_cp_commit();
}

template <int R>
__device__ __forceinline__ void tgx_gemv_cpasync(const unsigned* __restrict__ w, const float* __restrict__ wscale,
                                                 const signed char* __restrict__ xq, const float* __restrict__ xs,
                                                 const int* __restrict__ xsum, const half_t* __restrict__ residual,
                                                 half_t* __restrict__ out, int N, int K, int act) {
    signed char* x_s = reinterpret_cast<signed char*>(tgx_smem);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    uint4* ring = reinterpret_cast<uint4*>(x_s + K) + (size_t)warp * TGX_CP_STAGES * R * 32;
    const int KW = K >> 4;
    const int chunks = (KW + TGX_CHUNK_WORDS - 1) / TGX_CHUNK_WORDS;
    const int row0 = (blockIdx.x * TGX_CP_WARPS + warp) * R;

    tgx_stage_acts(x_s, xq, K);
    tgx_cp_commit();  // group 0: activations

#pragma unroll
    for (int c = 0; c < TGX_CP_STAGES - 1; ++c) tgx_cp_issue<R>(w, ring, row0, N, KW, chunks, c, lane);
    tgx_cp_wait<TGX_CP_STAGES - 1>();  // the activation group (oldest) is complete for this thread
    __syncthreads();                   // ... and for every thread

    int acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0;
    for (int c = 0; c < chunks; ++c) {
        tgx_cp_issue<R>(w, ring, row0, N, KW, chunks, c + TGX_CP_STAGES - 1, lane);
        tgx_cp_wait<TGX_CP_STAGES - 1>();  // chunk c landed (this lane's copies)
        const int base = c * TGX_CHUNK_WORDS;
        const int L = min(TGX_CHUNK_WORDS, KW - base) >> 2;
        if (lane < L) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint4 wv = ring[((c % TGX_CP_STAGES) * R + r) * 32 + lane];
                acc[r] = tgx_chunk_dot(wv, x_s, base, L, lane, acc[r]);
            }
        }
    }
    tgx_cp_wait<0>();
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = tgx_warp_sum(acc[r]);
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row0 + r;
            if (row < N) tgx_epilogue(acc[r], xsum[0], xs[0], wscale[row], act, residual, out, row);
        }
    }
}

#define TGX_DEFINE_CPASYNC(R)                                                                                  \
    extern "C" __global__ void __launch_bounds__(TGX_CP_WARPS * 32, 1)                                          \
        tgx_gemv_cpasync_r##R(const unsigned* __restrict__ w, const float* __restrict__ wscale,                \
                              const signed char* __restrict__ xq, const float* __restrict__ xs,                \
                              const int* __restrict__ xsum, const half_t* __restrict__ residual,               \
                              half_t* __restrict__ out, int N, int K, int act) {                               \
        tgx_gemv_cpasync<R>(w, wscale, xq, xs, xsum, residual, out, N, K, act);                                \
    }
TGX_DEFINE_CPASYNC(1)
TGX_DEFINE_CPASYNC(2)
TGX_DEFINE_CPASYNC(4)

// =================================================================================================
// 2. Split-K. grid = (ceil(N / (warps * 2)), splits). Split s covers chunks [s*per, (s+1)*per).
//    Partial int32 sums go to part[N] with atomicAdd (integer, so the order cannot change the
//    result); `part` must be zeroed before the launch. tgx_splitk_finalize applies the epilogue.
// =================================================================================================
extern "C" __global__ void __launch_bounds__(256)
    tgx_gemv_splitk_r2(const unsigned* __restrict__ w, const signed char* __restrict__ xq, int* __restrict__ part,
                       int N, int K) {
    const int lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * 2;
    if (row0 >= N) return;
    const int KW = K >> 4;
    const int chunks = (KW + TGX_CHUNK_WORDS - 1) / TGX_CHUNK_WORDS;
    const int per = (chunks + gridDim.y - 1) / gridDim.y;
    const int c0 = blockIdx.y * per, c1 = min(chunks, c0 + per);
    int acc[2] = {0, 0};
    for (int c = c0; c < c1; ++c) {
        const int base = c * TGX_CHUNK_WORDS;
        const int L = min(TGX_CHUNK_WORDS, KW - base) >> 2;
        if (lane < L) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = min(row0 + r, N - 1);
                const uint4 wv = __ldg(reinterpret_cast<const uint4*>(w + (size_t)row * KW + base) + lane);
                acc[r] = tgx_chunk_dot(wv, xq, base, L, lane, acc[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) acc[r] = tgx_warp_sum(acc[r]);
    if (lane == 0) {
        atomicAdd(&part[row0], acc[0]);
        if (row0 + 1 < N) atomicAdd(&part[row0 + 1], acc[1]);
    }
}

extern "C" __global__ void tgx_splitk_finalize(const int* __restrict__ part, const float* __restrict__ wscale,
                                               const float* __restrict__ xs, const int* __restrict__ xsum,
                                               const half_t* __restrict__ residual, half_t* __restrict__ out, int N,
                                               int act) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) tgx_epilogue(part[row], xsum[0], xs[0], wscale[row], act, residual, out, row);
}

// =================================================================================================
// 3. LUT GEMV (natural layout). For each pair of activations (x[2m], x[2m+1]) build the 16-entry
//    table T[c0 + 4*c1] = c0*x[2m] + c1*x[2m+1] over codes c in {0..3} (code 3 never occurs). One
//    weight nibble is then one table index: 8 lookups per 32-bit word instead of 4 dp4a. Tables
//    are built per K tile of TGX_LUT_KT weights (32 KiB of int16 in smem) and shared by the
//    block's TGX_LUT_ROWS rows. On GPUs with dp4a this is a reference point rather than a win
//    (dp4a does 4 MACs per instruction); it is the formulation LUT-based CPU/FPGA engines use.
//    Block = 256 threads (8 warps x 8 rows); K % 16 == 0.
// =================================================================================================
#define TGX_LUT_KT 2048
#define TGX_LUT_ROWS 64

extern "C" __global__ void __launch_bounds__(256)
    tgx_gemv_lut(const unsigned* __restrict__ w_nat, const float* __restrict__ wscale,
                 const signed char* __restrict__ xq, const float* __restrict__ xs, const int* __restrict__ xsum,
                 const half_t* __restrict__ residual, half_t* __restrict__ out, int N, int K, int act) {
    __shared__ short lut[(TGX_LUT_KT / 2) * 16];  // 1024 pairs x 16 entries x 2 B = 32 KiB
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int KW = K >> 4;
    const int rows_per_warp = TGX_LUT_ROWS / 8;
    const int row0 = blockIdx.x * TGX_LUT_ROWS + warp * rows_per_warp;
    int acc[TGX_LUT_ROWS / 8];
#pragma unroll
    for (int r = 0; r < rows_per_warp; ++r) acc[r] = 0;

    for (int k0 = 0; k0 < K; k0 += TGX_LUT_KT) {
        const int kt = min(TGX_LUT_KT, K - k0);
        __syncthreads();  // previous tile's lookups are done
        for (int e = threadIdx.x; e < (kt / 2) * 16; e += blockDim.x) {
            const int pair = e >> 4, idx = e & 15;
            const int c0 = idx & 3, c1 = idx >> 2;
            lut[e] = (short)(c0 * (int)xq[k0 + 2 * pair] + c1 * (int)xq[k0 + 2 * pair + 1]);
        }
        __syncthreads();
        const int wt = kt >> 4;  // words in this tile
#pragma unroll
        for (int r = 0; r < rows_per_warp; ++r) {
            const int row = min(row0 + r, N - 1);
            const unsigned* wr = w_nat + (size_t)row * KW + (k0 >> 4);
            int a = acc[r];
            for (int wi = lane; wi < wt; wi += 32) {
                const unsigned word = __ldg(wr + wi);
                const short* t = lut + wi * 8 * 16;  // 8 pairs per word
#pragma unroll
                for (int m = 0; m < 8; ++m) a += t[m * 16 + ((word >> (4 * m)) & 15u)];
            }
            acc[r] = a;
        }
    }
#pragma unroll
    for (int r = 0; r < rows_per_warp; ++r) acc[r] = tgx_warp_sum(acc[r]);
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < rows_per_warp; ++r) {
            const int row = row0 + r;
            if (row < N) tgx_epilogue(acc[r], xsum[0], xs[0], wscale[row], act, residual, out, row);
        }
    }
}

// =================================================================================================
// 4. Warp-specialized GEMV. Warp 0 is the producer: it streams the block's weight tiles (8 rows x
//    one 128-word chunk = 4 KiB) into a 2-stage smem ring with cp.async. Warps 1..4 consume: two
//    rows each, dp4a against activations staged in smem. Hand-off uses named barriers (ids 1..4,
//    id 0 is __syncthreads): FULL[s] = 1 + s (producer arrives, consumers sync) and
//    EMPTY[s] = 3 + s (consumers arrive, producer syncs), all 160 threads counted.
//    Dynamic smem: K bytes of activations + 2 * 8 * 512 bytes of weights. Grid = ceil(N / 8).
// =================================================================================================
#define TGX_WS_CONSUMERS 4
#define TGX_WS_ROWS (TGX_WS_CONSUMERS * 2)
#define TGX_WS_THREADS ((TGX_WS_CONSUMERS + 1) * 32)

extern "C" __global__ void __launch_bounds__(TGX_WS_THREADS)
    tgx_gemv_warpspec(const unsigned* __restrict__ w, const float* __restrict__ wscale,
                      const signed char* __restrict__ xq, const float* __restrict__ xs, const int* __restrict__ xsum,
                      const half_t* __restrict__ residual, half_t* __restrict__ out, int N, int K, int act) {
    signed char* x_s = reinterpret_cast<signed char*>(tgx_smem);
    uint4* ring = reinterpret_cast<uint4*>(x_s + K);  // [2][TGX_WS_ROWS][32] uint4
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int KW = K >> 4;
    const int chunks = (KW + TGX_CHUNK_WORDS - 1) / TGX_CHUNK_WORDS;
    const int rowb = blockIdx.x * TGX_WS_ROWS;

    tgx_stage_acts(x_s, xq, K);
    tgx_cp_commit();
    tgx_cp_wait<0>();
    __syncthreads();

    if (warp == 0) {
        for (int c = 0; c < chunks; ++c) {
            const int s = c & 1;
            if (c >= 2) {  // wait until the consumers released stage s
                if (s == 0) tgx_bar_sync<3, TGX_WS_THREADS>();
                else tgx_bar_sync<4, TGX_WS_THREADS>();
            }
            const int base = c * TGX_CHUNK_WORDS;
            const int L = min(TGX_CHUNK_WORDS, KW - base) >> 2;
            if (lane < L) {
#pragma unroll
                for (int r = 0; r < TGX_WS_ROWS; ++r) {
                    const int row = min(rowb + r, N - 1);
                    tgx_cp_async16(&ring[(s * TGX_WS_ROWS + r) * 32 + lane],
                                   reinterpret_cast<const uint4*>(w + (size_t)row * KW + base) + lane);
                }
            }
            tgx_cp_commit();
            tgx_cp_wait<0>();
            __threadfence_block();
            if (s == 0) tgx_bar_arrive<1, TGX_WS_THREADS>();  // stage s is full
            else tgx_bar_arrive<2, TGX_WS_THREADS>();
        }
    } else {
        const int r0 = (warp - 1) * 2;
        int acc[2] = {0, 0};
        for (int c = 0; c < chunks; ++c) {
            const int s = c & 1;
            if (s == 0) tgx_bar_sync<1, TGX_WS_THREADS>();
            else tgx_bar_sync<2, TGX_WS_THREADS>();
            const int base = c * TGX_CHUNK_WORDS;
            const int L = min(TGX_CHUNK_WORDS, KW - base) >> 2;
            if (lane < L) {
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const uint4 wv = ring[(s * TGX_WS_ROWS + r0 + r) * 32 + lane];
                    acc[r] = tgx_chunk_dot(wv, x_s, base, L, lane, acc[r]);
                }
            }
            // Release stage s only if the producer will wait for it (it syncs EMPTY from chunk 2 on).
            if (c + 2 < chunks) {
                if (s == 0) tgx_bar_arrive<3, TGX_WS_THREADS>();
                else tgx_bar_arrive<4, TGX_WS_THREADS>();
            }
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) acc[r] = tgx_warp_sum(acc[r]);
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = rowb + r0 + r;
                if (row < N) tgx_epilogue(acc[r], xsum[0], xs[0], wscale[row], act, residual, out, row);
            }
        }
    }
}
