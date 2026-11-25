//! Nasdaq TotalView-ITCH 5.0 replay: zero-copy message parser, per-symbol limit order book and a
//! small top-of-book feature extractor for signal models ([`crate::mlp`]).
//!
//! This is the workload behind `docs/ITCH_REPLAY.md` and the `itch` binary: replay a public
//! Nasdaq ITCH sample, rebuild the book, and score every book update with a small MLP, timing
//! parse / book / features / inference separately.
//!
//! Framing: the public `*.NASDAQ_ITCH50` files are a sequence of `[u16 big-endian length][message]`
//! records (the "binary file" framing Nasdaq ships, not MoldUDP64). All integers are big-endian;
//! timestamps are 6-byte nanoseconds since midnight (Eastern); prices are `u32` with 4 implied
//! decimals (so one cent = 100).

use std::collections::{BTreeMap, HashMap};
use std::hash::{BuildHasherDefault, Hasher};

/// One cent in ITCH price units (4 implied decimals).
pub const TICK: u32 = 100;
/// 09:30:00 and 16:00:00 Eastern in nanoseconds since midnight.
pub const RTH_OPEN_NS: u64 = 34_200 * 1_000_000_000;
pub const RTH_CLOSE_NS: u64 = 57_600 * 1_000_000_000;
/// Features per snapshot; the model input is the current snapshot plus the previous one.
pub const SNAP: usize = 32;
pub const FEATURES: usize = 2 * SNAP;
/// Levels summed for the depth features.
const DEPTH_LEVELS: usize = 5;

/// Message length (excluding the 2-byte frame) for every ITCH 5.0 message type, from the spec.
pub fn msg_len(t: u8) -> Option<usize> {
    Some(match t {
        b'S' => 12,
        b'R' => 39,
        b'H' => 25,
        b'Y' => 20,
        b'L' => 26,
        b'V' => 35,
        b'W' => 12,
        b'K' => 28,
        b'J' => 35,
        b'h' => 21,
        b'A' => 36,
        b'F' => 40,
        b'E' => 31,
        b'C' => 36,
        b'X' => 23,
        b'D' => 19,
        b'U' => 35,
        b'P' => 44,
        b'Q' => 40,
        b'B' => 19,
        b'I' => 50,
        b'N' => 20,
        b'O' => 48,
        _ => return None,
    })
}

#[inline(always)]
fn be16(b: &[u8], o: usize) -> u16 {
    u16::from_be_bytes([b[o], b[o + 1]])
}
#[inline(always)]
fn be32(b: &[u8], o: usize) -> u32 {
    u32::from_be_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
#[inline(always)]
fn be64(b: &[u8], o: usize) -> u64 {
    u64::from_be_bytes(b[o..o + 8].try_into().unwrap())
}
#[inline(always)]
fn be48(b: &[u8], o: usize) -> u64 {
    ((be16(b, o) as u64) << 32) | be32(b, o + 2) as u64
}

/// A decoded message. Only the fields the book needs are kept; everything else is `Other`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Msg {
    /// `S`: system event code (`O` start of messages, `Q` market open, `M` close, `C` end ...).
    System {
        ts: u64,
        code: u8,
    },
    /// `R`: stock directory: maps `locate` to an 8-byte space-padded symbol.
    StockDir {
        locate: u16,
        stock: [u8; 8],
    },
    /// `A` / `F`: add order (F carries an MPID attribution, ignored here).
    Add {
        locate: u16,
        ts: u64,
        oref: u64,
        buy: bool,
        shares: u32,
        price: u32,
    },
    /// `E`: executed at the resting price.
    Exec {
        locate: u16,
        ts: u64,
        oref: u64,
        shares: u32,
    },
    /// `C`: executed at a different price (shares still come off the resting order).
    ExecPx {
        locate: u16,
        ts: u64,
        oref: u64,
        shares: u32,
        price: u32,
    },
    /// `X`: partial cancel.
    Cancel {
        locate: u16,
        ts: u64,
        oref: u64,
        shares: u32,
    },
    /// `D`: delete.
    Delete {
        locate: u16,
        ts: u64,
        oref: u64,
    },
    /// `U`: replace (delete `old`, add `new` on the same side with new shares/price).
    Replace {
        locate: u16,
        ts: u64,
        old: u64,
        new: u64,
        shares: u32,
        price: u32,
    },
    /// `P`: non-displayed (hidden) trade; does not touch the visible book.
    Trade {
        locate: u16,
        ts: u64,
        buy: bool,
        shares: u32,
        price: u32,
    },
    Other {
        typ: u8,
        locate: u16,
        ts: u64,
    },
}

impl Msg {
    pub fn locate(&self) -> u16 {
        match *self {
            Msg::System { .. } => 0,
            Msg::StockDir { locate, .. }
            | Msg::Add { locate, .. }
            | Msg::Exec { locate, .. }
            | Msg::ExecPx { locate, .. }
            | Msg::Cancel { locate, .. }
            | Msg::Delete { locate, .. }
            | Msg::Replace { locate, .. }
            | Msg::Trade { locate, .. }
            | Msg::Other { locate, .. } => locate,
        }
    }
}

/// Parse one message body (no frame). `None` for an unknown type or a short buffer.
#[inline]
pub fn parse(b: &[u8]) -> Option<Msg> {
    let t = *b.first()?;
    if b.len() < msg_len(t)? {
        return None;
    }
    let locate = be16(b, 1);
    let ts = be48(b, 5);
    Some(match t {
        b'S' => Msg::System { ts, code: b[11] },
        b'R' => Msg::StockDir {
            locate,
            stock: b[11..19].try_into().unwrap(),
        },
        b'A' | b'F' => Msg::Add {
            locate,
            ts,
            oref: be64(b, 11),
            buy: b[19] == b'B',
            shares: be32(b, 20),
            price: be32(b, 32),
        },
        b'E' => Msg::Exec {
            locate,
            ts,
            oref: be64(b, 11),
            shares: be32(b, 19),
        },
        b'C' => Msg::ExecPx {
            locate,
            ts,
            oref: be64(b, 11),
            shares: be32(b, 19),
            price: be32(b, 32),
        },
        b'X' => Msg::Cancel {
            locate,
            ts,
            oref: be64(b, 11),
            shares: be32(b, 19),
        },
        b'D' => Msg::Delete {
            locate,
            ts,
            oref: be64(b, 11),
        },
        b'U' => Msg::Replace {
            locate,
            ts,
            old: be64(b, 11),
            new: be64(b, 19),
            shares: be32(b, 27),
            price: be32(b, 31),
        },
        b'P' => Msg::Trade {
            locate,
            ts,
            buy: b[19] == b'B',
            shares: be32(b, 20),
            price: be32(b, 32),
        },
        _ => Msg::Other { typ: t, locate, ts },
    })
}

/// Iterate `(offset, body)` over a length-framed ITCH buffer. Stops at the first truncated frame.
pub struct Frames<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Frames<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
}

impl<'a> Iterator for Frames<'a> {
    type Item = &'a [u8];
    #[inline]
    fn next(&mut self) -> Option<&'a [u8]> {
        let p = self.pos;
        if p + 2 > self.buf.len() {
            return None;
        }
        let n = be16(self.buf, p) as usize;
        if p + 2 + n > self.buf.len() {
            return None;
        }
        self.pos = p + 2 + n;
        Some(&self.buf[p + 2..p + 2 + n])
    }
}

/// Multiplicative (Fx-style) hasher for order reference numbers: SipHash costs more than the
/// whole book update. Order refs are exchange-assigned, not attacker-chosen, in this replay.
#[derive(Default, Clone, Copy)]
pub struct RefHasher(u64);
impl Hasher for RefHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
        }
    }
    #[inline]
    fn write_u64(&mut self, v: u64) {
        self.0 = (self.0.rotate_left(5) ^ v).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}
type RefMap<V> = HashMap<u64, V, BuildHasherDefault<RefHasher>>;

#[derive(Debug, Clone, Copy)]
struct Order {
    price: u32,
    shares: u32,
    locate: u16,
    buy: bool,
}

/// Visible depth for one symbol: price -> resting shares.
#[derive(Debug, Default, Clone)]
pub struct SymBook {
    pub bids: BTreeMap<u32, u64>,
    pub asks: BTreeMap<u32, u64>,
}

impl SymBook {
    pub fn best_bid(&self) -> Option<(u32, u64)> {
        self.bids.iter().next_back().map(|(&p, &q)| (p, q))
    }
    pub fn best_ask(&self) -> Option<(u32, u64)> {
        self.asks.iter().next().map(|(&p, &q)| (p, q))
    }
    /// Add `delta` shares at `price`; a level that drops to zero (or below) is removed. One tree
    /// descent per call (the entry handle also does the removal).
    fn level(&mut self, buy: bool, price: u32, delta: i64) {
        use std::collections::btree_map::Entry;
        let side = if buy { &mut self.bids } else { &mut self.asks };
        match side.entry(price) {
            Entry::Occupied(mut e) => {
                let v = *e.get() as i64 + delta;
                if v <= 0 {
                    e.remove();
                } else {
                    *e.get_mut() = v as u64;
                }
            }
            Entry::Vacant(e) => {
                if delta > 0 {
                    e.insert(delta as u64);
                }
            }
        }
    }
}

/// What a message did to a tracked symbol's book (input to the feature extractor).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Add,
    Cancel,
    Exec,
    Replace,
    HiddenTrade,
}

#[derive(Debug, Clone, Copy)]
pub struct Update {
    pub locate: u16,
    pub ts: u64,
    pub kind: Kind,
    /// Side of the resting order (for executions: the passive side).
    pub buy: bool,
    pub shares: u32,
    pub price: u32,
}

/// Order-by-order book for every symbol whose `locate` is tracked.
pub struct Book {
    orders: RefMap<Order>,
    /// Indexed by stock locate; `None` = not tracked.
    pub syms: Vec<Option<SymBook>>,
    /// Messages that referenced an unknown order (should stay 0 on a complete feed).
    pub unknown_refs: u64,
}

impl Book {
    /// Track the given locates (from the stock directory). An empty slice tracks every symbol.
    pub fn new(track: &[u16]) -> Self {
        let mut syms = vec![None; 65536];
        if track.is_empty() {
            syms.iter_mut().for_each(|s| *s = Some(SymBook::default()));
        } else {
            for &l in track {
                syms[l as usize] = Some(SymBook::default());
            }
        }
        let mut orders = RefMap::default();
        orders.reserve(1 << 20);
        Self {
            orders,
            syms,
            unknown_refs: 0,
        }
    }

    pub fn sym(&self, locate: u16) -> Option<&SymBook> {
        self.syms[locate as usize].as_ref()
    }

    /// Apply one message. Returns the change for a tracked symbol's visible book (or a hidden
    /// trade print), `None` otherwise.
    #[inline]
    pub fn apply(&mut self, m: &Msg) -> Option<Update> {
        match *m {
            Msg::Add {
                locate,
                ts,
                oref,
                buy,
                shares,
                price,
            } => {
                let sb = self.syms[locate as usize].as_mut()?;
                sb.level(buy, price, shares as i64);
                self.orders.insert(
                    oref,
                    Order {
                        price,
                        shares,
                        locate,
                        buy,
                    },
                );
                Some(Update {
                    locate,
                    ts,
                    kind: Kind::Add,
                    buy,
                    shares,
                    price,
                })
            }
            Msg::Exec {
                locate,
                ts,
                oref,
                shares,
            }
            | Msg::ExecPx {
                locate,
                ts,
                oref,
                shares,
                ..
            } => {
                self.syms[locate as usize].as_ref()?;
                self.reduce(oref, shares, ts, Kind::Exec)
            }
            Msg::Cancel {
                locate,
                ts,
                oref,
                shares,
            } => {
                self.syms[locate as usize].as_ref()?;
                self.reduce(oref, shares, ts, Kind::Cancel)
            }
            Msg::Delete { locate, ts, oref } => {
                self.syms[locate as usize].as_ref()?;
                self.reduce(oref, u32::MAX, ts, Kind::Cancel)
            }
            Msg::Replace {
                locate,
                ts,
                old,
                new,
                shares,
                price,
            } => {
                self.syms[locate as usize].as_ref()?;
                let Some(o) = self.orders.remove(&old) else {
                    self.unknown_refs += 1;
                    return None;
                };
                let sb = self.syms[o.locate as usize].as_mut()?;
                sb.level(o.buy, o.price, -(o.shares as i64));
                sb.level(o.buy, price, shares as i64);
                self.orders.insert(
                    new,
                    Order {
                        price,
                        shares,
                        locate: o.locate,
                        buy: o.buy,
                    },
                );
                Some(Update {
                    locate,
                    ts,
                    kind: Kind::Replace,
                    buy: o.buy,
                    shares,
                    price,
                })
            }
            Msg::Trade {
                locate,
                ts,
                buy,
                shares,
                price,
            } => {
                self.syms[locate as usize].as_ref()?;
                Some(Update {
                    locate,
                    ts,
                    kind: Kind::HiddenTrade,
                    buy,
                    shares,
                    price,
                })
            }
            _ => None,
        }
    }

    #[inline]
    fn reduce(&mut self, oref: u64, shares: u32, ts: u64, kind: Kind) -> Option<Update> {
        let Some(o) = self.orders.get_mut(&oref) else {
            self.unknown_refs += 1;
            return None;
        };
        let take = shares.min(o.shares);
        o.shares -= take;
        let (locate, buy, price, left) = (o.locate, o.buy, o.price, o.shares);
        if left == 0 {
            self.orders.remove(&oref);
        }
        let sb = self.syms[locate as usize].as_mut()?;
        sb.level(buy, price, -(take as i64));
        Some(Update {
            locate,
            ts,
            kind,
            buy,
            shares: take,
            price,
        })
    }

    pub fn live_orders(&self) -> usize {
        self.orders.len()
    }
}

/// Per-symbol state for the feature extractor (all O(1) per update, no allocation).
#[derive(Debug, Clone)]
pub struct FeatState {
    prev_bid: (u32, u64),
    prev_ask: (u32, u64),
    ofi_fast: f32,
    ofi_slow: f32,
    flow_fast: f32,
    flow_slow: f32,
    dt_ewma: f32,
    last_ts: u64,
    mids: [u32; 128],
    n: usize,
    prev_snap: [f32; SNAP],
}

impl Default for FeatState {
    fn default() -> Self {
        Self {
            prev_bid: (0, 0),
            prev_ask: (0, 0),
            ofi_fast: 0.0,
            ofi_slow: 0.0,
            flow_fast: 0.0,
            flow_slow: 0.0,
            dt_ewma: 0.0,
            last_ts: 0,
            mids: [0; 128],
            n: 0,
            prev_snap: [0.0; SNAP],
        }
    }
}

#[inline(always)]
fn slog(v: f32) -> f32 {
    v.signum() * v.abs().ln_1p()
}

/// Entries of the `ln(1 + i)` table: share counts, queue sizes and level counts are integers and
/// mostly below this (on the 2019-01-30 slice about 90% of touch sizes are under 2^12).
const LN1P_TABLE: usize = 1 << 15;

#[inline]
fn ln1p_table() -> &'static [f32] {
    static T: std::sync::OnceLock<Box<[f32]>> = std::sync::OnceLock::new();
    T.get_or_init(|| (0..LN1P_TABLE).map(|i| (i as f32).ln_1p()).collect())
}

/// `v.ln_1p()`, bit for bit, with a table lookup when `v` is a small non-negative integer (the
/// table holds exactly `(i as f32).ln_1p()`; the bit comparison keeps `-0.0` and non-integers on
/// the libm path).
#[inline(always)]
fn ln_1p_int(v: f32) -> f32 {
    let i = v as usize;
    if i < LN1P_TABLE && (i as f32).to_bits() == v.to_bits() {
        ln1p_table()[i]
    } else {
        v.ln_1p()
    }
}

/// [`slog`] through [`ln_1p_int`] (for integer-valued arguments such as the order-flow
/// imbalance).
#[inline(always)]
fn slog_int(v: f32) -> f32 {
    v.signum() * ln_1p_int(v.abs())
}

/// A decision point: a tracked symbol's book changed during regular trading hours and both sides
/// are present and uncrossed.
#[derive(Debug, Clone, Copy)]
pub struct Decision {
    pub locate: u16,
    pub ts: u64,
    /// Mid in ITCH units x2 (bid + ask), integral so it compares exactly.
    pub mid2: u32,
}

/// Compute the raw (unnormalized) feature vector for `u` into `out[..FEATURES]`.
/// Returns `None` when this update is not a decision point (outside RTH, one-sided or crossed).
#[inline]
pub fn features(sb: &SymBook, st: &mut FeatState, u: &Update, out: &mut [f32]) -> Option<Decision> {
    // One descent per side: the first level of each iterator is the touch, the next four give
    // the depth features (the reference walks each tree twice).
    let mut bids = sb.bids.iter().rev();
    let mut asks = sb.asks.iter();
    let (bid, ask) = (bids.next(), asks.next());
    let dt = u.ts.saturating_sub(st.last_ts);
    st.last_ts = u.ts;
    let (Some((&b0, &b1)), Some((&a0, &a1))) = (bid, ask) else {
        return None;
    };
    let (b, a) = ((b0, b1), (a0, a1));
    if a.0 <= b.0 {
        return None;
    }
    // Order-flow imbalance (Cont, Kukanov & Stoikov 2014) from the change in the touch.
    let (pb, pa) = (st.prev_bid, st.prev_ask);
    let e_b = if b.0 > pb.0 {
        b.1 as f32
    } else if b.0 == pb.0 {
        b.1 as f32 - pb.1 as f32
    } else {
        -(pb.1 as f32)
    };
    let e_a = if a.0 < pa.0 {
        a.1 as f32
    } else if a.0 == pa.0 {
        a.1 as f32 - pa.1 as f32
    } else {
        -(pa.1 as f32)
    };
    let ofi = if pb.0 == 0 { 0.0 } else { e_b - e_a };
    st.prev_bid = b;
    st.prev_ask = a;
    st.ofi_fast += 0.3 * (ofi - st.ofi_fast);
    st.ofi_slow += 0.03 * (ofi - st.ofi_slow);
    // Signed executed volume: an execution against a resting bid is seller-initiated.
    let flow = match u.kind {
        Kind::Exec | Kind::HiddenTrade => {
            let s = u.shares as f32;
            if u.buy {
                -s
            } else {
                s
            }
        }
        _ => 0.0,
    };
    st.flow_fast += 0.3 * (flow - st.flow_fast);
    st.flow_slow += 0.03 * (flow - st.flow_slow);
    let dt_us = dt as f32 * 1e-3;
    let dt_log = dt_us.ln_1p();
    st.dt_ewma += 0.05 * (dt_log - st.dt_ewma);

    let in_rth = u.ts >= RTH_OPEN_NS && u.ts < RTH_CLOSE_NS;
    let mid2 = b.0 + a.0;
    st.mids[st.n % 128] = mid2;
    st.n += 1;
    let ret = |lag: usize| -> f32 {
        if st.n > lag {
            (mid2 as f32 - st.mids[(st.n - 1 - lag) % 128] as f32) / (2.0 * TICK as f32)
        } else {
            0.0
        }
    };

    let (mut bd, mut ad) = ([0f32; DEPTH_LEVELS], [0f32; DEPTH_LEVELS]);
    let mut acc = b.1 as f32;
    bd[0] = acc;
    for (i, (_, &q)) in bids.take(DEPTH_LEVELS - 1).enumerate() {
        acc += q as f32;
        bd[i + 1] = acc;
    }
    for i in 1..DEPTH_LEVELS {
        if bd[i] == 0.0 {
            bd[i] = bd[i - 1];
        }
    }
    acc = a.1 as f32;
    ad[0] = acc;
    for (i, (_, &q)) in asks.take(DEPTH_LEVELS - 1).enumerate() {
        acc += q as f32;
        ad[i + 1] = acc;
    }
    for i in 1..DEPTH_LEVELS {
        if ad[i] == 0.0 {
            ad[i] = ad[i - 1];
        }
    }
    let imb = |x: f32, y: f32| (x - y) / (x + y).max(1.0);
    let spread = (a.0 - b.0) as f32 / TICK as f32;
    let (bq, aq) = (b.1 as f32, a.1 as f32);
    let micro = (b.0 as f32 * aq + a.0 as f32 * bq) / (aq + bq).max(1.0);
    let mid = mid2 as f32 * 0.5;

    let mut f = [0f32; SNAP];
    f[0] = spread;
    f[1] = imb(bq, aq);
    f[2] = imb(bd[1], ad[1]);
    f[3] = imb(bd[2], ad[2]);
    f[4] = imb(bd[4], ad[4]);
    f[5] = ln_1p_int(bq);
    f[6] = ln_1p_int(aq);
    f[7] = ln_1p_int(bd[4]);
    f[8] = ln_1p_int(ad[4]);
    f[9] = (micro - mid) / TICK as f32;
    f[10] = slog_int(ofi);
    f[11] = slog(st.ofi_fast);
    f[12] = slog(st.ofi_slow);
    f[13] = slog(st.flow_fast);
    f[14] = slog(st.flow_slow);
    f[15] = ret(1);
    f[16] = ret(10);
    f[17] = ret(100);
    f[18] = dt_log;
    f[19] = st.dt_ewma;
    let k = match u.kind {
        Kind::Add => 20,
        Kind::Cancel => 21,
        Kind::Exec => 22,
        Kind::Replace => 23,
        Kind::HiddenTrade => 24,
    };
    f[k] = 1.0;
    f[25] = if u.buy { 1.0 } else { -1.0 };
    // Distance of the event from its own side's touch, in ticks (0 = at the touch).
    let touch = if u.buy { b.0 } else { a.0 };
    f[26] = ((u.price as f32 - touch as f32) / TICK as f32)
        .abs()
        .min(50.0);
    f[27] = ln_1p_int(u.shares as f32);
    f[28] = (u.ts.saturating_sub(RTH_OPEN_NS)) as f32 / (RTH_CLOSE_NS - RTH_OPEN_NS) as f32;
    f[29] = ln_1p_int(sb.bids.len() as f32);
    f[30] = ln_1p_int(sb.asks.len() as f32);
    f[31] = 0.0;

    out[..SNAP].copy_from_slice(&f);
    out[SNAP..FEATURES].copy_from_slice(&st.prev_snap);
    st.prev_snap = f;
    if !in_rth {
        return None;
    }
    Some(Decision {
        locate: u.locate,
        ts: u.ts,
        mid2,
    })
}

/// The original straightforward feature extractor, kept as the bit-exactness oracle for
/// [`features`] (tests and `itch replay` compare the two).
#[doc(hidden)]
/// Returns `None` when this update is not a decision point (outside RTH, one-sided or crossed).
#[inline]
pub fn features_reference(
    sb: &SymBook,
    st: &mut FeatState,
    u: &Update,
    out: &mut [f32],
) -> Option<Decision> {
    let (bid, ask) = (sb.best_bid(), sb.best_ask());
    let dt = u.ts.saturating_sub(st.last_ts);
    st.last_ts = u.ts;
    let (Some(b), Some(a)) = (bid, ask) else {
        return None;
    };
    if a.0 <= b.0 {
        return None;
    }
    // Order-flow imbalance (Cont, Kukanov & Stoikov 2014) from the change in the touch.
    let (pb, pa) = (st.prev_bid, st.prev_ask);
    let e_b = if b.0 > pb.0 {
        b.1 as f32
    } else if b.0 == pb.0 {
        b.1 as f32 - pb.1 as f32
    } else {
        -(pb.1 as f32)
    };
    let e_a = if a.0 < pa.0 {
        a.1 as f32
    } else if a.0 == pa.0 {
        a.1 as f32 - pa.1 as f32
    } else {
        -(pa.1 as f32)
    };
    let ofi = if pb.0 == 0 { 0.0 } else { e_b - e_a };
    st.prev_bid = b;
    st.prev_ask = a;
    st.ofi_fast += 0.3 * (ofi - st.ofi_fast);
    st.ofi_slow += 0.03 * (ofi - st.ofi_slow);
    // Signed executed volume: an execution against a resting bid is seller-initiated.
    let flow = match u.kind {
        Kind::Exec | Kind::HiddenTrade => {
            let s = u.shares as f32;
            if u.buy {
                -s
            } else {
                s
            }
        }
        _ => 0.0,
    };
    st.flow_fast += 0.3 * (flow - st.flow_fast);
    st.flow_slow += 0.03 * (flow - st.flow_slow);
    let dt_us = dt as f32 * 1e-3;
    st.dt_ewma += 0.05 * (dt_us.ln_1p() - st.dt_ewma);

    let in_rth = u.ts >= RTH_OPEN_NS && u.ts < RTH_CLOSE_NS;
    let mid2 = b.0 + a.0;
    st.mids[st.n % 128] = mid2;
    st.n += 1;
    let ret = |lag: usize| -> f32 {
        if st.n > lag {
            (mid2 as f32 - st.mids[(st.n - 1 - lag) % 128] as f32) / (2.0 * TICK as f32)
        } else {
            0.0
        }
    };

    let (mut bd, mut ad) = ([0f32; DEPTH_LEVELS], [0f32; DEPTH_LEVELS]);
    let mut acc = 0f32;
    for (i, (_, &q)) in sb.bids.iter().rev().take(DEPTH_LEVELS).enumerate() {
        acc += q as f32;
        bd[i] = acc;
    }
    for i in 1..DEPTH_LEVELS {
        if bd[i] == 0.0 {
            bd[i] = bd[i - 1];
        }
    }
    acc = 0.0;
    for (i, (_, &q)) in sb.asks.iter().take(DEPTH_LEVELS).enumerate() {
        acc += q as f32;
        ad[i] = acc;
    }
    for i in 1..DEPTH_LEVELS {
        if ad[i] == 0.0 {
            ad[i] = ad[i - 1];
        }
    }
    let imb = |x: f32, y: f32| (x - y) / (x + y).max(1.0);
    let spread = (a.0 - b.0) as f32 / TICK as f32;
    let (bq, aq) = (b.1 as f32, a.1 as f32);
    let micro = (b.0 as f32 * aq + a.0 as f32 * bq) / (aq + bq).max(1.0);
    let mid = mid2 as f32 * 0.5;

    let mut f = [0f32; SNAP];
    f[0] = spread;
    f[1] = imb(bq, aq);
    f[2] = imb(bd[1], ad[1]);
    f[3] = imb(bd[2], ad[2]);
    f[4] = imb(bd[4], ad[4]);
    f[5] = bq.ln_1p();
    f[6] = aq.ln_1p();
    f[7] = bd[4].ln_1p();
    f[8] = ad[4].ln_1p();
    f[9] = (micro - mid) / TICK as f32;
    f[10] = slog(ofi);
    f[11] = slog(st.ofi_fast);
    f[12] = slog(st.ofi_slow);
    f[13] = slog(st.flow_fast);
    f[14] = slog(st.flow_slow);
    f[15] = ret(1);
    f[16] = ret(10);
    f[17] = ret(100);
    f[18] = dt_us.ln_1p();
    f[19] = st.dt_ewma;
    let k = match u.kind {
        Kind::Add => 20,
        Kind::Cancel => 21,
        Kind::Exec => 22,
        Kind::Replace => 23,
        Kind::HiddenTrade => 24,
    };
    f[k] = 1.0;
    f[25] = if u.buy { 1.0 } else { -1.0 };
    // Distance of the event from its own side's touch, in ticks (0 = at the touch).
    let touch = if u.buy { b.0 } else { a.0 };
    f[26] = ((u.price as f32 - touch as f32) / TICK as f32)
        .abs()
        .min(50.0);
    f[27] = (u.shares as f32).ln_1p();
    f[28] = (u.ts.saturating_sub(RTH_OPEN_NS)) as f32 / (RTH_CLOSE_NS - RTH_OPEN_NS) as f32;
    f[29] = (sb.bids.len() as f32).ln_1p();
    f[30] = (sb.asks.len() as f32).ln_1p();
    f[31] = 0.0;

    out[..SNAP].copy_from_slice(&f);
    out[SNAP..FEATURES].copy_from_slice(&st.prev_snap);
    st.prev_snap = f;
    if !in_rth {
        return None;
    }
    Some(Decision {
        locate: u.locate,
        ts: u.ts,
        mid2,
    })
}

/// Standardize with train-set statistics and clip to +-5 (the int8 activation quantizer of the
/// ternary model is per-sample absmax, so one outlier would crush the other 63 features).
#[inline]
pub fn normalize(x: &mut [f32], mean: &[f32], inv_std: &[f32]) {
    for i in 0..x.len() {
        x[i] = ((x[i] - mean[i]) * inv_std[i]).clamp(-5.0, 5.0);
    }
}

/// 8-byte space-padded ITCH symbol from a ticker.
pub fn symbol8(s: &str) -> [u8; 8] {
    let mut out = [b' '; 8];
    for (o, c) in out.iter_mut().zip(s.bytes()) {
        *o = c;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(body: &[u8]) -> Vec<u8> {
        let mut v = (body.len() as u16).to_be_bytes().to_vec();
        v.extend_from_slice(body);
        v
    }
    fn hdr(t: u8, locate: u16, ts: u64, len: usize) -> Vec<u8> {
        let mut b = vec![0u8; len];
        b[0] = t;
        b[1..3].copy_from_slice(&locate.to_be_bytes());
        b[5..11].copy_from_slice(&ts.to_be_bytes()[2..]);
        b
    }
    fn add(locate: u16, ts: u64, oref: u64, buy: bool, shares: u32, price: u32) -> Vec<u8> {
        let mut b = hdr(b'A', locate, ts, 36);
        b[11..19].copy_from_slice(&oref.to_be_bytes());
        b[19] = if buy { b'B' } else { b'S' };
        b[20..24].copy_from_slice(&shares.to_be_bytes());
        b[24..32].copy_from_slice(b"TEST    ");
        b[32..36].copy_from_slice(&price.to_be_bytes());
        b
    }

    #[test]
    fn parses_and_rebuilds_book() {
        let ts = RTH_OPEN_NS + 1;
        let mut buf = Vec::new();
        buf.extend(frame(&add(7, ts, 1, true, 100, 1_000_000)));
        buf.extend(frame(&add(7, ts, 2, true, 50, 999_900)));
        buf.extend(frame(&add(7, ts, 3, false, 200, 1_000_200)));
        let mut e = hdr(b'E', 7, ts, 31);
        e[11..19].copy_from_slice(&1u64.to_be_bytes());
        e[19..23].copy_from_slice(&40u32.to_be_bytes());
        buf.extend(frame(&e));
        let mut u = hdr(b'U', 7, ts, 35);
        u[11..19].copy_from_slice(&3u64.to_be_bytes());
        u[19..27].copy_from_slice(&4u64.to_be_bytes());
        u[27..31].copy_from_slice(&150u32.to_be_bytes());
        u[31..35].copy_from_slice(&1_000_100u32.to_be_bytes());
        buf.extend(frame(&u));
        let mut d = hdr(b'D', 7, ts, 19);
        d[11..19].copy_from_slice(&2u64.to_be_bytes());
        buf.extend(frame(&d));

        let mut book = Book::new(&[7]);
        let mut n = 0;
        for f in Frames::new(&buf) {
            let m = parse(f).unwrap();
            assert!(book.apply(&m).is_some());
            n += 1;
        }
        assert_eq!(n, 6);
        let sb = book.sym(7).unwrap();
        assert_eq!(sb.best_bid(), Some((1_000_000, 60)));
        assert_eq!(sb.best_ask(), Some((1_000_100, 150)));
        assert_eq!(sb.bids.len(), 1);
        assert_eq!(book.live_orders(), 2);
        assert_eq!(book.unknown_refs, 0);

        let mut st = FeatState::default();
        let mut x = [0f32; FEATURES];
        let up = Update {
            locate: 7,
            ts,
            kind: Kind::Add,
            buy: true,
            shares: 1,
            price: 1_000_000,
        };
        let d = features(sb, &mut st, &up, &mut x).unwrap();
        assert_eq!(d.mid2, 2_000_100);
        assert_eq!(x[0], 1.0); // one-cent spread
        assert!((x[1] - (60.0 - 150.0) / 210.0).abs() < 1e-6);
    }

    /// Random order flow on two symbols: the optimized `features` (single tree walk, ln(1+x)
    /// table) must equal `features_reference` bit for bit, and the book's price levels must equal
    /// a naive sum over the live orders after every message.
    #[test]
    fn fast_features_and_book_match_reference_on_random_flow() {
        use rand::{rngs::StdRng, RngExt, SeedableRng};
        let mut rng = StdRng::seed_from_u64(5);
        let mut book = Book::new(&[3, 9]);
        let (mut st, mut st_ref) = (
            vec![FeatState::default(); 16],
            vec![FeatState::default(); 16],
        );
        let (mut x, mut x_ref) = ([0f32; FEATURES], [0f32; FEATURES]);
        // Live orders: oref -> (locate, buy, price, shares).
        let mut live: Vec<(u64, u16, bool, u32, u32)> = Vec::new();
        let mut next_ref = 1u64;
        let mut ts = RTH_OPEN_NS - 1_000_000;
        let mut decisions = 0;
        for step in 0..60_000 {
            ts += rng.random_range(1..2_000_000u64);
            let locate = if rng.random() { 3 } else { 9 };
            let mid = if locate == 3 { 1_000_000u32 } else { 250_000 };
            let pick = rng.random_range(0..10u32);
            let m = if live.is_empty() || pick < 4 {
                let buy: bool = rng.random();
                let off = rng.random_range(0..8u32) * TICK;
                let price = if buy { mid - TICK - off } else { mid + off };
                // Mostly round lots, sometimes large (beyond the ln table) or odd lots.
                let shares = match rng.random_range(0..10u32) {
                    0 => rng.random_range(40_000..200_000u32),
                    1 => rng.random_range(1..100u32),
                    _ => 100 * rng.random_range(1..20u32),
                };
                live.push((next_ref, locate, buy, price, shares));
                next_ref += 1;
                Msg::Add {
                    locate,
                    ts,
                    oref: next_ref - 1,
                    buy,
                    shares,
                    price,
                }
            } else {
                let i = rng.random_range(0..live.len());
                let (oref, loc, buy, price, shares) = live[i];
                match pick {
                    4 | 5 => {
                        let take = rng.random_range(1..=shares);
                        if take == shares {
                            live.swap_remove(i);
                        } else {
                            live[i].4 -= take;
                        }
                        Msg::Exec {
                            locate: loc,
                            ts,
                            oref,
                            shares: take,
                        }
                    }
                    6 => {
                        let take = rng.random_range(1..=shares);
                        if take == shares {
                            live.swap_remove(i);
                        } else {
                            live[i].4 -= take;
                        }
                        Msg::Cancel {
                            locate: loc,
                            ts,
                            oref,
                            shares: take,
                        }
                    }
                    7 => {
                        live.swap_remove(i);
                        Msg::Delete {
                            locate: loc,
                            ts,
                            oref,
                        }
                    }
                    8 => {
                        let np = if buy { price - TICK } else { price + TICK };
                        let ns = 100 * rng.random_range(1..10u32);
                        live[i] = (next_ref, loc, buy, np, ns);
                        next_ref += 1;
                        Msg::Replace {
                            locate: loc,
                            ts,
                            old: oref,
                            new: next_ref - 1,
                            shares: ns,
                            price: np,
                        }
                    }
                    _ => Msg::Trade {
                        locate: loc,
                        ts,
                        buy,
                        shares: 100,
                        price,
                    },
                }
            };
            let Some(u) = book.apply(&m) else {
                panic!("step {step}: tracked message produced no update");
            };
            let sb = book.sym(u.locate).unwrap();
            if step % 97 == 0 {
                for (l, side_buy) in [(3u16, true), (3, false), (9, true), (9, false)] {
                    let mut want = std::collections::BTreeMap::new();
                    for &(_, loc, buy, price, shares) in &live {
                        if loc == l && buy == side_buy {
                            *want.entry(price).or_insert(0u64) += shares as u64;
                        }
                    }
                    let b = book.sym(l).unwrap();
                    let got = if side_buy { &b.bids } else { &b.asks };
                    assert_eq!(got, &want, "step {step} locate {l} buy {side_buy}");
                }
            }
            let li = u.locate as usize;
            let d = features(sb, &mut st[li], &u, &mut x);
            let d_ref = features_reference(sb, &mut st_ref[li], &u, &mut x_ref);
            assert_eq!(d.map(|d| (d.ts, d.mid2)), d_ref.map(|d| (d.ts, d.mid2)));
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&x), bits(&x_ref), "step {step}");
            decisions += d.is_some() as usize;
        }
        assert!(decisions > 10_000, "only {decisions} decisions");
        assert_eq!(book.unknown_refs, 0);
    }

    #[test]
    fn ln_1p_table_is_bit_exact() {
        for v in [
            0.0f32,
            -0.0,
            1.0,
            2.5,
            100.0,
            32767.0,
            32768.0,
            1e6,
            f32::NAN,
            -3.0,
        ] {
            assert_eq!(ln_1p_int(v).to_bits(), v.ln_1p().to_bits(), "{v}");
        }
        for i in 0..LN1P_TABLE {
            let v = i as f32;
            assert_eq!(ln_1p_int(v).to_bits(), v.ln_1p().to_bits(), "{i}");
        }
    }

    #[test]
    fn untracked_symbols_are_ignored() {
        let mut book = Book::new(&[1]);
        let m = parse(&add(2, 0, 9, true, 10, 100)).unwrap();
        assert!(book.apply(&m).is_none());
        assert_eq!(book.live_orders(), 0);
    }

    #[test]
    fn every_known_type_has_a_length() {
        for t in b"SRHYLVWKJhAFECXDUPQBINO" {
            assert!(msg_len(*t).is_some());
        }
        assert!(msg_len(b'Z').is_none());
    }
}
