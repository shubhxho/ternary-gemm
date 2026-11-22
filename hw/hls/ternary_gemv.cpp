#include "ternary_gemv.h"

int32_t ternary_dot16(uint32_t codes, const int8_t* q16) {
#pragma HLS INLINE
    int32_t s = 0;
    for (int t = 0; t < 16; ++t) {
#pragma HLS UNROLL
        const uint32_t c = (codes >> (2 * t)) & 3u;
        const int32_t a = q16[t];
        // 0 -> -a, 1 -> 0, 2 -> +a, 3 (invalid) -> 0: a mux, no multiplier.
        s += (c == 2u) ? a : (c == 0u) ? -a : 0;
    }
    return s;
}

void ternary_gemv(const uint32_t* w, const int8_t* q, int32_t* acc, int rows, int kw) {
#pragma HLS INTERFACE m_axi port = w offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = q offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = acc offset = slave bundle = gmem2
#pragma HLS INTERFACE s_axilite port = rows
#pragma HLS INTERFACE s_axilite port = kw
#pragma HLS INTERFACE s_axilite port = return
rows_loop:
    for (int r = 0; r < rows; ++r) {
        int32_t s = 0;
    words_loop:
        for (int j = 0; j < kw; ++j) {
#pragma HLS PIPELINE II = 1
            s += ternary_dot16(w[r * kw + j], q + 16 * j);
        }
        acc[r] = s;
    }
}
