//! A tiny fork-join pool for microsecond-scale work: workers spin on an epoch counter instead of
//! sleeping, so handing out a job costs a cache-line round trip rather than a futex wake-up
//! (`std::thread::scope` and rayon both pay several microseconds per fork at this scale).
//!
//! The trade-off is that idle workers keep their cores busy (they spin with `spin_loop` hints,
//! then fall back to `yield_now` after [`SPIN_LIMIT`] polls). Use it only where a few cores can be
//! dedicated to latency, e.g. a pinned trading process.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

/// Polls before a waiting worker starts yielding its time slice.
pub const SPIN_LIMIT: u32 = 1 << 16;

type Job = *const (dyn Fn(usize) + Sync);

/// One cache line (two on CPUs whose adjacent-line prefetcher pairs them) per hot atomic, so the
/// owner's epoch bump and each worker's completion flag never share a line.
#[repr(align(128))]
struct Padded<T>(T);

struct Shared {
    /// Bumped by the owner to publish a job; read (spun on) by every worker.
    epoch: Padded<AtomicU64>,
    /// Per-worker "finished epoch e" flags, written only by their worker (no contended RMW).
    done: Box<[Padded<AtomicU64>]>,
    stop: AtomicBool,
    job: UnsafeCell<Option<Job>>,
}

// SAFETY: `job` is written only by the owner while no job is in flight (every worker has stored
// the previous epoch into its `done` flag), and read by workers only after observing the epoch
// bump (Acquire/Release).
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

/// Fork-join pool with `threads - 1` spinning workers; the caller is worker 0.
pub struct SpinPool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    threads: usize,
}

impl std::fmt::Debug for SpinPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpinPool")
            .field("threads", &self.threads)
            .finish()
    }
}

impl SpinPool {
    /// A pool running jobs on `threads` threads in total (the caller included). `threads <= 1`
    /// spawns nothing and runs jobs inline.
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let shared = Arc::new(Shared {
            epoch: Padded(AtomicU64::new(0)),
            done: (0..threads).map(|_| Padded(AtomicU64::new(0))).collect(),
            stop: AtomicBool::new(false),
            job: UnsafeCell::new(None),
        });
        let workers = (1..threads)
            .map(|id| {
                let sh = shared.clone();
                std::thread::Builder::new()
                    .name(format!("tgemm-spin-{id}"))
                    .spawn(move || worker(sh, id))
                    .expect("spawn spin worker")
            })
            .collect();
        Self {
            shared,
            workers,
            threads,
        }
    }

    /// Total threads a job runs on.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Run `f(i)` for every `i in 0..threads()` (the caller runs `i = 0`) and wait for all.
    pub fn run<F: Fn(usize) + Sync>(&self, f: F) {
        if self.threads == 1 {
            return f(0);
        }
        let dynf: &(dyn Fn(usize) + Sync) = &f;
        // SAFETY: the job pointer is only dereferenced until every worker has bumped `done` for
        // this epoch, and we do not return (so `f` stays alive) before that.
        let ptr: Job = unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), Job>(dynf) };
        unsafe { *self.shared.job.get() = Some(ptr) };
        // Only the owner writes the epoch, so a plain load + store is enough.
        let e = self.shared.epoch.0.load(Ordering::Relaxed) + 1;
        self.shared.epoch.0.store(e, Ordering::Release);
        f(0);
        for d in &self.shared.done[1..] {
            let mut spins = 0u32;
            while d.0.load(Ordering::Acquire) != e {
                spins = spins.saturating_add(1);
                if spins < SPIN_LIMIT {
                    std::hint::spin_loop();
                } else {
                    std::thread::yield_now();
                }
            }
        }
        unsafe { *self.shared.job.get() = None };
    }
}

fn worker(sh: Arc<Shared>, id: usize) {
    let mut seen = 0u64;
    loop {
        let mut spins = 0u32;
        let e = loop {
            let e = sh.epoch.0.load(Ordering::Acquire);
            if e != seen {
                break e;
            }
            spins = spins.saturating_add(1);
            if spins < SPIN_LIMIT {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        };
        seen = e;
        if sh.stop.load(Ordering::Acquire) {
            return;
        }
        // SAFETY: published before the epoch bump we just observed; alive until `done` reaches
        // the worker count (see `SpinPool::run`).
        let job = unsafe { (*sh.job.get()).expect("job published") };
        unsafe { (*job)(id) };
        sh.done[id].0.store(e, Ordering::Release);
    }
}

impl Drop for SpinPool {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        self.shared.epoch.0.fetch_add(1, Ordering::Release);
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn every_index_runs_once_per_job() {
        for threads in [1usize, 2, 4] {
            let pool = SpinPool::new(threads);
            let hits: Vec<AtomicUsize> = (0..threads).map(|_| AtomicUsize::new(0)).collect();
            for _ in 0..1000 {
                pool.run(|i| {
                    hits[i].fetch_add(1, Ordering::Relaxed);
                });
            }
            for h in &hits {
                assert_eq!(h.load(Ordering::Relaxed), 1000);
            }
        }
    }

    #[test]
    fn disjoint_writes_are_visible_after_run() {
        let pool = SpinPool::new(3);
        let out: Vec<AtomicUsize> = (0..300).map(|_| AtomicUsize::new(0)).collect();
        for round in 1..50 {
            pool.run(|t| {
                for i in (t..300).step_by(3) {
                    out[i].store(round * i, Ordering::Relaxed);
                }
            });
            for (i, v) in out.iter().enumerate() {
                assert_eq!(v.load(Ordering::Relaxed), round * i);
            }
        }
    }
}
