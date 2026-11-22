## What

<!-- One logical change. -->

## Verification

- [ ] `make check` (fmt, clippy -D warnings, tests, rustdoc)
- [ ] Kernel changes: `make compile-check` register/spill diff attached
- [ ] Bit-exact against the CPU reference (which test covers it?)
- [ ] Hardware used for any numbers quoted (or "CPU only"); nothing theoretical presented as measured
