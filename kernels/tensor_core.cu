// ternary-gemm: tensor-core path for ternary GEMM/GEMV (W2A8 / W2A16).
//
// Compiled at runtime with NVRTC for sm_80+ (see src/gpu.rs). Self-contained: no #include,
// so NVRTC needs no CUDA headers on disk. This file is also meant to be compiled with nvcc:
//
//   nvcc -std=c++17 -ptx -arch=compute_80  -o tensor_core_80.ptx  kernels/tensor_core.cu
//   nvcc -std=c++17 -ptx -arch=compute_89  -o tensor_core_89.ptx  kernels/tensor_core.cu
//   nvcc -std=c++17 -ptx -arch=compute_120 -o tensor_core_120.ptx kernels/tensor_core.cu
//
// ---------------------------------------------------------------------------------
// Design (TriRun / BitBLAS W2A8 style)
//   * Weights stay in the *natural* 2-bit packed layout: 16 codes per u32, code c = w + 1
//     in {0,1,2}, weight k at word k/16, bits 2*(k % 16). (The dp4a kernels use the
//     permuted layout; this path unpacks from the natural layout already stored on device.)
//   * Each warp owns a 16-row output tile (the M dimension of m16n8k16). Batch is tiled
//     by 8 (the N dimension). K is reduced in steps of 16.
//   * Each k-tile, the warp unpacks one natural word per owned row into int8 or fp16
//     *A* fragments in registers, and packs the matching activation slice into *B*
//     fragments (activations are staged in shared memory so every warp in the block
//     reads the same K-tile once).
//   * The MMA is PTX `mma.sync.aligned.m16n8k16` (sm_80+, including sm_89 and sm_120):
//       W2A8:  .row.col.s32.s8.s8.s32
//       W2A16: .row.col.f32.f16.f16.f32
//   * Epilogue is the same as the reference: y = acc * row_scale / act_scale (+ residual)
//     for W2A8, and y = acc * row_scale (+ residual) for W2A16. Because weights are
//     decoded to {-1,0,+1} before the MMA, there is no (code-1) / xsum correction.
//
// The 8x vs fp16 figure at batch 1 is a *theoretical bandwidth ceiling* (2-bit vs 16-bit
// weight traffic), not a measured GPU result. Prior art that reached near 8x on GEMV:
// TriRun (github.com/NolanoOrg/TriRun) and Microsoft BitBLAS.
// ---------------------------------------------------------------------------------

typedef unsigned short half_t;

__device__ __forceinline__ float tc_h2f(half_t h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
__device__ __forceinline__ half_t tc_f2h(float f) {
    half_t h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}
__device__ __forceinline__ unsigned tc_pack_f16x2(float a, float b) {
    return (unsigned)tc_f2h(a) | ((unsigned)tc_f2h(b) << 16);
}

__device__ __forceinline__ void tc_store(half_t* __restrict__ out, const half_t* __restrict__ residual,
                                         size_t idx, float v) {
    if (residual) v = __fadd_rn(v, tc_h2f(residual[idx]));
    out[idx] = tc_f2h(v);
}

// Pack 4 consecutive ternary weights (codes at bits 2*(t0..t0+3)) as signed bytes, w = c - 1.
__device__ __forceinline__ unsigned tc_pack_w4(unsigned word, int t0) {
    unsigned r = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int w = (int)((word >> (2 * (t0 + i))) & 3u) - 1;
        r |= (unsigned)(w & 0xff) << (8 * i);
    }
    return r;
}


// Pack two consecutive ternary weights at k-indices t0, t0+1 of a natural word as one f16x2 register.
__device__ __forceinline__ unsigned tc_pack_f16_pair(unsigned word, int t0) {
    return tc_pack_f16x2((float)((int)((word >> (2 * t0)) & 3u) - 1),
                         (float)((int)((word >> (2 * (t0 + 1))) & 3u) - 1));
}

// m16n8k16 int8 MMA: A is 2x u32 (8 s8), B is 1x u32 (4 s8), C/D are 4x s32.
__device__ __forceinline__ void mma_m16n8k16_s8(int& d0, int& d1, int& d2, int& d3, unsigned a0,
                                                unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                 "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
                 : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                 : "r"(a0), "r"(a1), "r"(b0), "r"(d0), "r"(d1), "r"(d2), "r"(d3));
}

// m16n8k16 fp16 MMA: A is 4x u32 (8 f16), B is 2x u32 (4 f16), C/D are 4x f32.
__device__ __forceinline__ void mma_m16n8k16_f16(float& d0, float& d1, float& d2, float& d3,
                                                 unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                                 unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(d0), "f"(d1), "f"(d2),
                   "f"(d3));
}

// Load the natural word for output row `row` at k-block `kb`, or the all-zero-weight word.
__device__ __forceinline__ unsigned tc_load_word(const unsigned* __restrict__ w_nat, int row, int N,
                                                 int KW, int kb) {
    if (row >= N) return 0x55555555u;  // code 1 == weight 0
    return __ldg(w_nat + (size_t)row * (size_t)KW + (size_t)kb);
}

// ---------------------------------------------------------------------------------
// W2A8: 2-bit weights unpacked to int8 fragments, int8 activations, s32 accumulate.
// grid.x = ceil(N / (16 * warps_per_block)), block.x = 32 * warps_per_block (1..8 warps).
// ---------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256)
    ternary_tc_i8(const unsigned* __restrict__ w_nat, const float* __restrict__ wscale,
                  const signed char* __restrict__ xq, const float* __restrict__ xs,
                  const half_t* __restrict__ residual, half_t* __restrict__ out, int N, int K,
                  int batch) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const int row_base = (blockIdx.x * warps + wid) * 16;
    if (row_base >= N) return;

    const int group = lane >> 2;     // 0..7  -> MMA row and MMA-N column
    const int tidg = lane & 3;       // 0..3  -> K nibble inside the 16-wide tile
    const int KW = K >> 4;
    const int ntiles = (batch + 7) >> 3;  // 1 or 2 (batch <= 16)

    // Activations for this K-tile, all batch rows: [16 batch][16 K], 256 B.
    __shared__ signed char xsmem[16 * 16];

    int acc[2][4];
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[t][i] = 0;

    for (int kb = 0; kb < KW; ++kb) {
        const int k0 = kb << 4;
        // Cooperative load of the 16-wide K-tile of activations (one 16-byte row per batch).
        for (int i = threadIdx.x; i < 16 * 16; i += blockDim.x) {
            const int b = i >> 4, kk = i & 15;
            signed char v = 0;
            if (b < batch) v = xq[(size_t)b * (size_t)K + (size_t)(k0 + kk)];
            xsmem[i] = v;
        }
        __syncthreads();

        const unsigned word0 = tc_load_word(w_nat, row_base + group, N, KW, kb);
        const unsigned word1 = tc_load_word(w_nat, row_base + group + 8, N, KW, kb);
        const unsigned a0 = tc_pack_w4(word0, tidg * 4);
        const unsigned a1 = tc_pack_w4(word1, tidg * 4);

#pragma unroll
        for (int t = 0; t < 2; ++t) {
            if (t < ntiles) {
                const int b = t * 8 + group;
                unsigned b0 = 0;
                if (b < batch) {
                    const signed char* row = xsmem + b * 16 + tidg * 4;
                    b0 = (unsigned)(unsigned char)row[0] | ((unsigned)(unsigned char)row[1] << 8) |
                         ((unsigned)(unsigned char)row[2] << 16) | ((unsigned)(unsigned char)row[3] << 24);
                }
                mma_m16n8k16_s8(acc[t][0], acc[t][1], acc[t][2], acc[t][3], a0, a1, b0);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int t = 0; t < 2; ++t) {
        if (t >= ntiles) continue;
        // C fragment: (d0,d1,d2,d3) -> rows (group, group+8, group, group+8),
        //             cols (tidg*2, tidg*2, tidg*2+1, tidg*2+1) of the 16x8 tile.
        const int row_a = row_base + group;
        const int row_b = row_base + group + 8;
        const int col0 = t * 8 + tidg * 2;
        const int col1 = col0 + 1;
        const int vals[4] = {acc[t][0], acc[t][1], acc[t][2], acc[t][3]};
        const int rows[4] = {row_a, row_b, row_a, row_b};
        const int cols[4] = {col0, col0, col1, col1};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (rows[i] < N && cols[i] < batch) {
                float s = xs[cols[i]];
                float y = __fmul_rn(__fdiv_rn((float)vals[i], s), wscale[rows[i]]);
                tc_store(out, residual, (size_t)cols[i] * (size_t)N + (size_t)rows[i], y);
            }
        }
    }
}

// ---------------------------------------------------------------------------------
// W2A16: 2-bit weights unpacked to fp16 fragments, fp16 activations, fp32 accumulate.
// ---------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256)
    ternary_tc_f16(const unsigned* __restrict__ w_nat, const float* __restrict__ wscale,
                   const half_t* __restrict__ x, const half_t* __restrict__ residual,
                   half_t* __restrict__ out, int N, int K, int batch) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const int row_base = (blockIdx.x * warps + wid) * 16;
    if (row_base >= N) return;

    const int group = lane >> 2;
    const int tidg = lane & 3;
    const int KW = K >> 4;
    const int ntiles = (batch + 7) >> 3;

    // [16 batch][16 K] fp16 values, stored as 16-bit patterns.
    __shared__ half_t xsmem[16 * 16];

    float acc[2][4];
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[t][i] = 0.f;

    for (int kb = 0; kb < KW; ++kb) {
        const int k0 = kb << 4;
        for (int i = threadIdx.x; i < 16 * 16; i += blockDim.x) {
            const int b = i >> 4, kk = i & 15;
            half_t v = 0;
            if (b < batch) v = x[(size_t)b * (size_t)K + (size_t)(k0 + kk)];
            xsmem[i] = v;
        }
        __syncthreads();

        const unsigned word0 = tc_load_word(w_nat, row_base + group, N, KW, kb);
        const unsigned word1 = tc_load_word(w_nat, row_base + group + 8, N, KW, kb);
        // A fragment: 8 fp16 = 4 registers. See PTX m16n8k16.row.col.f16 layout.
        // a0,a1: row=group,   k = tidg*2 + {0,1} and tidg*2 + 8 + {0,1}
        // a2,a3: row=group+8, same k
        const unsigned a0 = tc_pack_f16_pair(word0, tidg * 2);
        const unsigned a1 = tc_pack_f16_pair(word0, tidg * 2 + 8);
        const unsigned a2 = tc_pack_f16_pair(word1, tidg * 2);
        const unsigned a3 = tc_pack_f16_pair(word1, tidg * 2 + 8);

#pragma unroll
        for (int t = 0; t < 2; ++t) {
            if (t < ntiles) {
                const int b = t * 8 + group;
                unsigned b0 = 0, b1 = 0;
                if (b < batch) {
                    const half_t* row = xsmem + b * 16;
                    const int k_lo = tidg * 2;
                    b0 = (unsigned)row[k_lo] | ((unsigned)row[k_lo + 1] << 16);
                    b1 = (unsigned)row[k_lo + 8] | ((unsigned)row[k_lo + 9] << 16);
                }
                mma_m16n8k16_f16(acc[t][0], acc[t][1], acc[t][2], acc[t][3], a0, a1, a2, a3, b0, b1);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int t = 0; t < 2; ++t) {
        if (t >= ntiles) continue;
        const int row_a = row_base + group;
        const int row_b = row_base + group + 8;
        const int col0 = t * 8 + tidg * 2;
        const int col1 = col0 + 1;
        const float vals[4] = {acc[t][0], acc[t][1], acc[t][2], acc[t][3]};
        const int rows[4] = {row_a, row_b, row_a, row_b};
        const int cols[4] = {col0, col0, col1, col1};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (rows[i] < N && cols[i] < batch) {
                float y = __fmul_rn(vals[i], wscale[rows[i]]);
                tc_store(out, residual, (size_t)cols[i] * (size_t)N + (size_t)rows[i], y);
            }
        }
    }
}
