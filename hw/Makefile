# FPGA skeleton: ternary dot / MAC RTL + testbench, HLS C++ reference.
#   make -C hw vectors   # regenerate test vectors from the Rust crate
#   make -C hw iverilog  # simulate with Icarus Verilog
#   make -C hw verilator # lint + simulate with Verilator (--binary, timing)
#   make -C hw hls       # build and run the HLS C++ kernel's software testbench (g++)
#   make -C hw all       # everything
VEC   := vectors
RTL   := rtl/ternary_dot16.v rtl/ternary_mac.v
TB    := tb/tb_ternary_mac.v
BUILD := build
CARGO ?= cargo

.PHONY: all vectors iverilog verilator lint hls clean

all: vectors iverilog verilator hls

vectors: $(VEC)/w.hex

$(VEC)/w.hex:
	cd .. && $(CARGO) run --quiet --release --example hw_vectors -- hw/$(VEC)

iverilog: vectors
	mkdir -p $(BUILD)
	iverilog -g2012 -Wall -I $(VEC) -o $(BUILD)/tb_iverilog $(RTL) $(TB)
	cd $(VEC) && vvp -n ../$(BUILD)/tb_iverilog

lint:
	verilator --lint-only -Wall --top-module ternary_mac $(RTL)

verilator: vectors lint
	mkdir -p $(BUILD)
	verilator --binary -Wall -Wno-DECLFILENAME -Wno-UNUSEDSIGNAL -Wno-BLKSEQ -Wno-INITIALDLY \
		-I$(VEC) --top-module tb_ternary_mac -Mdir $(BUILD)/verilator -o tb $(RTL) $(TB)
	cd $(VEC) && ../$(BUILD)/verilator/tb

hls: vectors
	mkdir -p $(BUILD)
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-label -Werror -o $(BUILD)/hls_tb hls/ternary_gemv.cpp hls/ternary_gemv_tb.cpp
	cd $(VEC) && ../$(BUILD)/hls_tb

clean:
	rm -rf $(BUILD)
