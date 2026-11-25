//! Checks ternary.zig against the Rust reference vectors (`langs/vectors/case*.txt`, written by
//! `cargo run --release --example lang_vectors`) and times the largest case.
//!
//! Usage: zig run -O ReleaseFast -mcpu=native langs/zig/check.zig -- langs/vectors

const std = @import("std");
const t = @import("ternary.zig");

const Case = struct {
    n: usize,
    k: usize,
    s: f32,
    scale: []f32,
    words: []u32,
    q: []i8,
    acc: []i32,
    out: []u16,
};

fn parseCase(gpa: std.mem.Allocator, text: []const u8) !Case {
    var it = std.mem.tokenizeAny(u8, text, " \n");
    const n = try std.fmt.parseInt(usize, it.next().?, 10);
    const k = try std.fmt.parseInt(usize, it.next().?, 10);
    const s: f32 = @bitCast(try std.fmt.parseInt(u32, it.next().?, 16));
    _ = it.next(); // sum(q): not needed for the w-domain accumulation
    const c = Case{
        .n = n,
        .k = k,
        .s = s,
        .scale = try gpa.alloc(f32, n),
        .words = try gpa.alloc(u32, n * k / 16),
        .q = try gpa.alloc(i8, k),
        .acc = try gpa.alloc(i32, n),
        .out = try gpa.alloc(u16, n),
    };
    for (c.scale) |*v| v.* = @bitCast(try std.fmt.parseInt(u32, it.next().?, 16));
    for (c.words) |*v| v.* = try std.fmt.parseInt(u32, it.next().?, 16);
    for (c.q) |*v| v.* = try std.fmt.parseInt(i8, it.next().?, 10);
    for (c.acc) |*v| v.* = try std.fmt.parseInt(i32, it.next().?, 10);
    for (c.out) |*v| v.* = try std.fmt.parseInt(u16, it.next().?, 16);
    return c;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(gpa);
    const dir = if (args.len > 1) args[1] else "langs/vectors";
    var failures: usize = 0;
    var last: ?Case = null;
    var i: usize = 0;
    while (true) : (i += 1) {
        const path = try std.fmt.allocPrint(gpa, "{s}/case{d}.txt", .{ dir, i });
        const text = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 << 20)) catch break;
        const c = try parseCase(gpa, text);
        const acc = try gpa.alloc(i32, c.n);
        const qt = try gpa.alloc(i16, c.k);
        t.transposeActs(c.q, c.k / 256 * 256, qt);
        t.gemvNatural(c.words, c.q, qt, c.n, c.k, acc);
        var bad: usize = 0;
        for (0..c.n) |r| {
            const y: u16 = @bitCast(t.dequant(acc[r], c.s, c.scale[r]));
            if (acc[r] != c.acc[r] or y != c.out[r]) bad += 1;
        }
        std.debug.print("zig case{d} n={d} k={d}: {s}\n", .{ i, c.n, c.k, if (bad == 0) "OK (acc + fp16 bits exact)" else "MISMATCH" });
        failures += bad;
        last = c;
    }
    if (i == 0) {
        std.debug.print("no vectors in {s}; run `cargo run --release --example lang_vectors`\n", .{dir});
        return error.NoVectors;
    }
    if (last) |c| {
        const acc = try gpa.alloc(i32, c.n);
        const qt = try gpa.alloc(i16, c.k);
        const iters: usize = 2000;
        const t0 = std.Io.Clock.awake.now(io);
        for (0..iters) |_| {
            t.transposeActs(c.q, c.k / 256 * 256, qt); // timed: it is per-call work
            t.gemvNatural(c.words, c.q, qt, c.n, c.k, acc);
            std.mem.doNotOptimizeAway(acc.ptr);
        }
        const t1 = std.Io.Clock.awake.now(io);
        const ns: f64 = @floatFromInt(t1.nanoseconds - t0.nanoseconds);
        std.debug.print("zig gemvNatural {d}x{d}: {d:.2} us/call (mean of {d}, this machine)\n", .{ c.n, c.k, ns / 1e3 / @as(f64, @floatFromInt(iters)), iters });
    }
    if (failures != 0) return error.Mismatch;
}
