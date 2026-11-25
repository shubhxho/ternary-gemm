//! Ternary W2A8 GEMV in Zig, using portable `@Vector` SIMD (lowered to SSE/AVX2/AVX-512/NEON by
//! LLVM for whatever `-mcpu` you build for).
//!
//! Weights use the ternary-gemm "natural" layout: weight `c` of a row lives in word `c / 16`,
//! bits `2 * (c % 16)`, as the code `w + 1` in {0, 1, 2}. Activations are int8 (per-token absmax,
//! scale `s`). Output `y[r] = fp16(acc[r] / s * row_scale[r])` with `acc[r] = sum_c w[r][c] q[c]`,
//! exactly the Rust reference (`reference::gemm_i8`): integer accumulation, one IEEE f32 divide,
//! one f32 multiply, round-to-nearest-even to f16.

const std = @import("std");

const lanes = 16;
const V32 = @Vector(lanes, i32);
const V16 = @Vector(lanes, i16);
const VU = @Vector(lanes, u32);

/// Bit offsets of the 16 codes inside a packed word.
const shifts: @Vector(lanes, u5) = .{ 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30 };

/// Activations regrouped for [`gemvNatural`]: `qt[g][j][i] = q[16 * (16 g + i) + j]`, i.e. for
/// each group of 16 words, weight slot `j` of word `i`. Built once per activation vector.
pub fn transposeActs(q: []const i8, k: usize, qt: []i16) void {
    std.debug.assert(k % 256 == 0 and qt.len >= k);
    for (0..k / 256) |g| {
        for (0..16) |j| {
            for (0..16) |i| qt[g * 256 + j * 16 + i] = q[16 * (16 * g + i) + j];
        }
    }
}

/// `acc[r] = sum_c w[r][c] * q[c]` for `r < n`.
///
/// SIMD across words rather than within one: 16 consecutive packed words form one u32 vector,
/// and weight slot `j` of all 16 is `(v >> 2j) & 3` with a *uniform* shift (one `vpsrld imm`), so
/// no per-lane variable shifts are needed. The activations are pre-grouped by [`transposeActs`]
/// to line up, and the multiply is replaced by compare + select (add/sub). `k % 256 == 0` for the
/// vector loop; a scalar tail handles the rest.
pub fn gemvNatural(words: []const u32, q: []const i8, qt: []const i16, n: usize, k: usize, acc: []i32) void {
    std.debug.assert(k % 16 == 0 and words.len >= n * k / 16 and q.len >= k and acc.len >= n);
    const kw = k / 16;
    const groups = kw / 16;
    const three: VU = @splat(3);
    const two: VU = @splat(2);
    const zero_u: VU = @splat(0);
    const zero: V32 = @splat(0);
    for (0..n) |r| {
        const row = words[r * kw ..][0..kw];
        var s0: V32 = @splat(0);
        var s1: V32 = @splat(0);
        for (0..groups) |g| {
            const v: VU = row[g * 16 ..][0..16].*;
            inline for (0..16) |j| {
                // Ternary needs no multiply: add q where w = +1 (code 2), subtract where w = -1
                // (code 0), i.e. a bit-sliced add/sub with two compares and two selects.
                const codes = (v >> @splat(@as(u5, 2 * j))) & three;
                const qv: V32 = @as(@Vector(lanes, i16), qt[g * 256 + j * 16 ..][0..16].*);
                const plus = @select(i32, codes == two, qv, zero);
                const minus = @select(i32, codes == zero_u, qv, zero);
                if (j % 2 == 0) {
                    s0 += plus - minus;
                } else {
                    s1 += plus - minus;
                }
            }
        }
        var total: i32 = @reduce(.Add, s0 + s1);
        for (groups * 16..kw) |w| {
            for (0..16) |j| {
                const code: i32 = @intCast((row[w] >> @intCast(2 * j)) & 3);
                total += (code - 1) * @as(i32, q[w * 16 + j]);
            }
        }
        acc[r] = total;
    }
}

/// Scalar oracle for the same layout.
pub fn gemvScalar(words: []const u32, q: []const i8, n: usize, k: usize, acc: []i32) void {
    const kw = k / 16;
    for (0..n) |r| {
        var s: i32 = 0;
        for (0..k) |c| {
            const code: i32 = @intCast((words[r * kw + c / 16] >> @intCast(2 * (c % 16))) & 3);
            s += (code - 1) * @as(i32, q[c]);
        }
        acc[r] = s;
    }
}

/// The reference epilogue: `fp16(acc / s * scale)`, f32 ops in this order, no FMA.
pub fn dequant(acc: i32, s: f32, scale: f32) f16 {
    var t: f32 = @floatFromInt(acc);
    t /= s;
    t *= scale;
    return @floatCast(t);
}

test "simd matches scalar on random data" {
    var prng = std.Random.DefaultPrng.init(42);
    const rnd = prng.random();
    const n = 37;
    const k = 1088;
    var words: [n * k / 16]u32 = undefined;
    for (&words) |*wd| {
        var v: u32 = 0;
        for (0..16) |i| v |= @as(u32, rnd.uintLessThan(u32, 3)) << @intCast(2 * i);
        wd.* = v;
    }
    var q: [k]i8 = undefined;
    for (&q) |*x| x.* = rnd.int(i8);
    var a: [n]i32 = undefined;
    var b: [n]i32 = undefined;
    var qt: [k]i16 = undefined;
    transposeActs(&q, k / 256 * 256, &qt);
    gemvNatural(&words, &q, &qt, n, k, &a);
    gemvScalar(&words, &q, n, k, &b);
    try std.testing.expectEqualSlices(i32, &b, &a);
}

test "dequant rounds like the reference" {
    // 1 / 3 * 3 in f32 is exactly 1.0; f16(1.0) = 0x3c00.
    try std.testing.expectEqual(@as(u16, 0x3c00), @as(u16, @bitCast(dequant(1, 3.0, 3.0))));
}
