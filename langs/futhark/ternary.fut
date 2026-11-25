-- Ternary W2A8 GEMV in Futhark: a purely functional, data-parallel description that the
-- Futhark compiler turns into sequential C, multicore C, OpenCL or CUDA.
--
-- Natural layout as in the Rust crate: weight 16*w + j of a row is (words[w] >> 2j) & 3 = w + 1.
-- acc[r] = sum_c w[r][c] * q[c] (exact i32), y[r] = f16((f32 acc / s) * scale[r]).
-- Floats cross the boundary as raw bits so the check against the Rust vectors is bit-exact.

def code (word: u32) (j: i64) : i32 = i32.u32 ((word >> u32.i64 (2 * j)) & 3)

-- Ternary weights need no multiply: +x for code 2, -x for code 0, 0 for code 1.
def tmul (c: i32) (x: i8) : i32 = if c == 2 then i32.i8 x else if c == 0 then -(i32.i8 x) else 0

entry gemv_acc [n][kw] (words: [n][kw]u32) (q: [kw][16]i8) : [n]i32 =
  map (\row ->
         i32.sum (map2 (\word qs -> i32.sum (map2 (\j x -> tmul (code word j) x) (iota 16) qs))
                       row q))
      words

entry gemv_bits [n][kw] (words: [n][kw]u32) (q: [kw][16]i8) (s_bits: u32) (scale_bits: [n]u32) : [n]u16 =
  let s = f32.from_bits s_bits
  in map2 (\a sb -> f16.to_bits (f16.f32 ((f32.i32 a / s) * f32.from_bits sb)))
          (gemv_acc words q) scale_bits
