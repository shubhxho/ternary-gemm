// HLS C++ reference of the ternary GEMV integer core (same contract as rtl/ternary_mac.v).
// Synthesizable with Vitis HLS (pragmas below); compiles as plain C++17 for the software testbench.
#pragma once
#include <cstdint>

// acc[r] = sum_k w[r][k] * q[k], weights in the crate's natural 2-bit layout (code = w + 1,
// weight t of a word at bits [2t+1:2t]), `kw` words (16 weights each) per row.
void ternary_gemv(const uint32_t* w, const int8_t* q, int32_t* acc, int rows, int kw);

// One 16-wide multiplier-free dot product (exposed for unit testing).
int32_t ternary_dot16(uint32_t codes, const int8_t* q16);
