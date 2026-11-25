# Ternary W2A8 GEMV in Mojo, with explicit SIMD vectors.
#
# Natural layout (as in the Rust crate): weight c of a row at word c // 16, bits 2 * (c % 16),
# stored as code w + 1. y[r] = fp16(acc[r] / s * row_scale[r]), acc[r] = sum_c w[r][c] * q[c].
# Checked bit for bit against the Rust reference vectors:
#   mojo run langs/mojo/ternary.mojo langs/vectors

from std.sys import argv
from std.time import perf_counter_ns
from std.memory import bitcast


def transpose_acts(q: List[Int8], k: Int, mut qt: List[Int16]):
    """Qt[g*256 + j*16 + i] = q[16*(16g + i) + j]: slot j of word i in each 16-word group."""
    for g in range(k // 256):
        for j in range(16):
            for i in range(16):
                qt[g * 256 + j * 16 + i] = Int16(q[16 * (16 * g + i) + j])


def gemv_natural(
    words: List[UInt32],
    q: List[Int8],
    qt: List[Int16],
    n: Int,
    k: Int,
    mut acc: List[Int32],
):
    """Acc[r] = sum_c w[r][c] * q[c].

    SIMD across 16 consecutive words: slot j of all 16 is (v >> 2j) & 3, a uniform shift, and
    the activations come pre-grouped from transpose_acts. The multiply is replaced by selects
    (add q where w = +1, subtract where w = -1). A scalar loop covers K % 256.
    """
    var kw = k // 16
    var groups = kw // 16
    var wp = words.unsafe_ptr()
    var tp = qt.unsafe_ptr()
    for r in range(n):
        var s0 = SIMD[DType.int32, 16](0)
        for g in range(groups):
            var v = wp.unsafe_load[width=16](r * kw + g * 16)
            comptime for j in range(16):
                var codes = (v >> UInt32(2 * j)) & 3
                var qv = tp.unsafe_load[width=16](g * 256 + j * 16).cast[DType.int32]()
                var zero = SIMD[DType.int32, 16](0)
                s0 += codes.eq(2).select(qv, zero) - codes.eq(0).select(qv, zero)
        var total = s0.reduce_add()
        for w in range(groups * 16, kw):
            var word = words[r * kw + w]
            for j in range(16):
                var code = Int32((word >> UInt32(2 * j)) & 3)
                total += (code - 1) * Int32(q[w * 16 + j])
        acc[r] = total


def hex32(t: StringSlice) -> UInt32:
    var v: UInt32 = 0
    for b in t.as_bytes():
        var d = UInt32(b)
        if d >= 97:
            d = d - 87  # a-f
        elif d >= 65:
            d = d - 55  # A-F
        else:
            d = d - 48  # 0-9
        v = (v << 4) | d
    return v


def dequant(acc: Int32, s: Float32, scale: Float32) -> Float16:
    var t = Float32(acc)
    t = t / s
    t = t * scale
    return t.cast[DType.float16]()


def main() raises:
    var dir = String("langs/vectors")
    var args = argv()
    if len(args) > 1:
        dir = String(args[1])
    var failures = 0
    var cases = 0
    for i in range(64):
        var path = dir + "/case" + String(i) + ".txt"
        var text: String
        try:
            with open(path, "r") as f:
                text = f.read()
        except:
            break
        var tok = text.split()
        var p = 0
        var n = Int(tok[p]); p += 1
        var k = Int(tok[p]); p += 1
        var s = bitcast[DType.float32](hex32(tok[p]))
        p += 2  # s, sum(q)
        var scale = List[Float32]()
        for _ in range(n):
            scale.append(bitcast[DType.float32](hex32(tok[p])))
            p += 1
        var words = List[UInt32]()
        for _ in range(n * k // 16):
            words.append(hex32(tok[p]))
            p += 1
        var q = List[Int8]()
        for _ in range(k):
            q.append(Int8(Int(tok[p])))
            p += 1
        var want_acc = List[Int32]()
        for _ in range(n):
            want_acc.append(Int32(Int(tok[p])))
            p += 1
        var want_out = List[UInt16]()
        for _ in range(n):
            want_out.append(UInt16(hex32(tok[p])))
            p += 1
        var acc = List[Int32](length=n, fill=0)
        var qt = List[Int16](length=k, fill=0)
        transpose_acts(q, k, qt)
        gemv_natural(words, q, qt, n, k, acc)
        var bad = 0
        for r in range(n):
            var y = bitcast[DType.uint16](dequant(acc[r], s, scale[r]))
            if acc[r] != want_acc[r] or y != want_out[r]:
                bad += 1
        print("mojo case", i, "n =", n, "k =", k, ":", "OK (acc + fp16 bits exact)" if bad == 0 else "MISMATCH")
        failures += bad
        cases += 1
        if i == 4:
            var iters = 500
            var t0 = perf_counter_ns()
            for _ in range(iters):
                transpose_acts(q, k, qt)  # per-call work, so it is timed
                gemv_natural(words, q, qt, n, k, acc)
            var t1 = perf_counter_ns()
            print("mojo gemv_natural", n, "x", k, ":", Float64(t1 - t0) / 1e3 / Float64(iters), "us/call (mean, this machine)")
    if cases == 0:
        raise Error("no vectors; run `cargo run --release --example lang_vectors`")
    if failures != 0:
        raise Error("mismatch")
