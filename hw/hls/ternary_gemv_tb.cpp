// Software testbench for the HLS kernel: same vectors as the RTL testbench (hw/vectors), each
// row has its own activation vector, so call the kernel one row at a time.
#include "ternary_gemv.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static std::vector<std::string> lines(const char* path) {
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "cannot open %s (run `make -C hw vectors`)\n", path);
        std::exit(2);
    }
    std::vector<std::string> v;
    for (std::string s; std::getline(f, s);)
        if (!s.empty()) v.push_back(s);
    return v;
}

int main() {
    const auto wl = lines("w.hex"), al = lines("a.hex"), el = lines("exp.hex");
    const size_t rows = el.size(), kw = wl.size() / rows;
    if (rows == 0 || wl.size() != rows * kw || al.size() != wl.size()) {
        std::fprintf(stderr, "inconsistent vector files\n");
        return 2;
    }
    size_t errors = 0;
    for (size_t r = 0; r < rows; ++r) {
        std::vector<uint32_t> w(kw);
        std::vector<int8_t> q(16 * kw);
        for (size_t j = 0; j < kw; ++j) {
            w[j] = static_cast<uint32_t>(std::stoul(wl[r * kw + j], nullptr, 16));
            const std::string& a = al[r * kw + j];  // 32 hex digits, activation 15 first
            for (int t = 0; t < 16; ++t) {
                const std::string byte = a.substr(2 * (15 - t), 2);
                q[16 * j + t] = static_cast<int8_t>(std::stoul(byte, nullptr, 16));
            }
        }
        int32_t acc = 0;
        ternary_gemv(w.data(), q.data(), &acc, 1, static_cast<int>(kw));
        const auto want = static_cast<int32_t>(std::stoul(el[r], nullptr, 16));
        if (acc != want && ++errors <= 10)
            std::printf("MISMATCH row %zu: got %d expected %d\n", r, acc, want);
    }
    if (errors) {
        std::printf("FAIL: %zu mismatches in %zu rows\n", errors, rows);
        return 1;
    }
    std::printf("PASS: HLS kernel, %zu rows x %zu words match the Rust reference\n", rows, kw);
    return 0;
}
