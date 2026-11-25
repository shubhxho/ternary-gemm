# Ternary W2A8 GEMV in Julia (no packages: Base SIMD via @simd / @inbounds and LLVM's vectorizer).
#
# Natural layout as in the Rust crate: weight c (0-based) of a row at word c ÷ 16, bits
# 2 (c % 16), code w + 1. y[r] = Float16(acc[r] / s * row_scale[r]) with an exact Int32 acc.
# Checked against the Rust reference vectors:
#   julia --project=@. -O3 langs/julia/ternary.jl langs/vectors

module Ternary

export gemv_natural!, transpose_acts!, dequant

"""
    transpose_acts!(qt, q)

Regroup activations so slot `j` of 16 consecutive words is contiguous:
`qt[g*256 + j*16 + i + 1] = q[16*(16g + i) + j + 1]` (0-based g, j, i).
"""
function transpose_acts!(qt::Vector{Int16}, q::Vector{Int8})
    k = length(q)
    @inbounds for g in 0:(k ÷ 256 - 1), j in 0:15, i in 0:15
        qt[g * 256 + j * 16 + i + 1] = q[16 * (16g + i) + j + 1]
    end
    return qt
end

"""
    gemv_natural!(acc, words, q, qt, n, k)

`acc[r] = Σ_c w[r][c] q[c]`. The inner loop runs across 16 consecutive words with a uniform
shift per slot (vectorizes to `vpsrld imm` + compares), and replaces the multiply with
`ifelse` add/sub, since every weight is -1, 0 or +1.
"""
function gemv_natural!(acc::Vector{Int32}, words::Vector{UInt32}, q::Vector{Int8},
                       qt::Vector{Int16}, n::Int, k::Int)
    kw = k ÷ 16
    groups = kw ÷ 16
    lanes = zeros(Int32, 16)  # per-lane partial sums: reduced once per row, not per slot
    @inbounds for r in 0:(n - 1)
        fill!(lanes, Int32(0))
        rb = r * kw
        for g in 0:(groups - 1), j in 0:15
            sh = UInt32(2j)
            @simd ivdep for i in 1:16
                c = (words[rb + g * 16 + i] >> sh) & 0x3
                x = Int32(qt[g * 256 + j * 16 + i])
                lanes[i] += ifelse(c == 0x2, x, Int32(0)) - ifelse(c == 0x0, x, Int32(0))
            end
        end
        total = sum(lanes)
        for w in (groups * 16):(kw - 1), j in 0:15
            c = Int32((words[rb + w + 1] >> UInt32(2j)) & 0x3)
            total += (c - Int32(1)) * Int32(q[w * 16 + j + 1])
        end
        acc[r + 1] = total
    end
    return acc
end

"The reference epilogue: Float16((Float32(acc) / s) * scale), round to nearest even."
dequant(acc::Int32, s::Float32, scale::Float32) = Float16((Float32(acc) / s) * scale)

end # module

using .Ternary

function load_case(path)
    tok = split(read(path, String))
    p = Ref(1)
    next() = (t = tok[p[]]; p[] += 1; t)
    n = parse(Int, next()); k = parse(Int, next())
    s = reinterpret(Float32, parse(UInt32, next(); base = 16)); next()
    scale = [reinterpret(Float32, parse(UInt32, next(); base = 16)) for _ in 1:n]
    words = [parse(UInt32, next(); base = 16) for _ in 1:(n * k ÷ 16)]
    q = [parse(Int8, next()) for _ in 1:k]
    acc = [parse(Int32, next()) for _ in 1:n]
    out = [parse(UInt16, next(); base = 16) for _ in 1:n]
    return (; n, k, s, scale, words, q, acc, out)
end

function main(dir)
    failures = 0
    last = nothing
    i = 0
    while isfile(joinpath(dir, "case$i.txt"))
        c = load_case(joinpath(dir, "case$i.txt"))
        qt = zeros(Int16, c.k)
        transpose_acts!(qt, c.q)
        acc = zeros(Int32, c.n)
        gemv_natural!(acc, c.words, c.q, qt, c.n, c.k)
        y = [reinterpret(UInt16, dequant(acc[r], c.s, c.scale[r])) for r in 1:c.n]
        ok = acc == c.acc && y == c.out
        println("julia case$i n=$(c.n) k=$(c.k): ", ok ? "OK (acc + fp16 bits exact)" : "MISMATCH")
        failures += !ok
        last = c
        i += 1
    end
    i == 0 && error("no vectors in $dir; run `cargo run --release --example lang_vectors`")
    c = last
    qt = zeros(Int16, c.k); acc = zeros(Int32, c.n)
    for _ in 1:50  # warm up (JIT)
        transpose_acts!(qt, c.q); gemv_natural!(acc, c.words, c.q, qt, c.n, c.k)
    end
    iters = 2000
    t0 = time_ns()
    for _ in 1:iters
        transpose_acts!(qt, c.q); gemv_natural!(acc, c.words, c.q, qt, c.n, c.k)
    end
    us = (time_ns() - t0) / 1e3 / iters
    println("julia gemv_natural! $(c.n)x$(c.k): ", round(us; digits = 2), " us/call (mean, this machine)")
    failures == 0 || exit(1)
end

main(length(ARGS) > 0 ? ARGS[1] : "langs/vectors")
