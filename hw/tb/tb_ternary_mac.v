// Testbench: streams the vectors from `cargo run --example hw_vectors` through ternary_mac and
// compares every row sum with the Rust-computed expectation. Prints PASS/FAIL and exits with
// $fatal on any mismatch (non-zero exit in both Icarus and Verilator).
`timescale 1ns/1ps
`default_nettype none
module tb_ternary_mac;
`include "params.vh"
    reg          clk = 0;
    reg          rst = 1;
    reg          in_valid = 0, first = 0, last = 0;
    reg  [31:0]  w_codes = 0;
    reg  [127:0] acts = 0;
    wire         acc_valid;
    wire signed [31:0] acc;

    reg [31:0]  wmem [0:ROWS*WORDS-1];
    reg [127:0] amem [0:ROWS*WORDS-1];
    reg [31:0]  emem [0:ROWS-1];

    ternary_mac dut (
        .clk(clk), .rst(rst), .in_valid(in_valid), .first(first), .last(last),
        .w_codes(w_codes), .acts(acts), .acc_valid(acc_valid), .acc(acc)
    );

    always #5 clk = ~clk;

    integer got = 0, errors = 0, r, j;

    always @(posedge clk) begin
        if (acc_valid) begin
            if (acc !== $signed(emem[got])) begin
                errors = errors + 1;
                if (errors <= 10)
                    $display("MISMATCH row %0d: got %0d expected %0d", got, acc, $signed(emem[got]));
            end
            got = got + 1;
        end
    end

    initial begin
        $readmemh("w.hex", wmem);
        $readmemh("a.hex", amem);
        $readmemh("exp.hex", emem);
        repeat (3) @(posedge clk);
        rst <= 0;
        @(posedge clk);
        // Back-to-back rows, one beat per cycle, no bubbles.
        for (r = 0; r < ROWS; r = r + 1) begin
            for (j = 0; j < WORDS; j = j + 1) begin
                in_valid <= 1;
                first    <= (j == 0);
                last     <= (j == WORDS - 1);
                w_codes  <= wmem[r*WORDS + j];
                acts     <= amem[r*WORDS + j];
                @(posedge clk);
            end
        end
        in_valid <= 0; first <= 0; last <= 0;
        repeat (5) @(posedge clk);
        if (got != ROWS) begin
            $display("FAIL: %0d of %0d rows produced", got, ROWS);
            $fatal(1);
        end
        if (errors != 0) begin
            $display("FAIL: %0d mismatches in %0d rows", errors, ROWS);
            $fatal(1);
        end
        $display("PASS: %0d rows x %0d words, all row sums match the Rust reference", ROWS, WORDS);
        $finish;
    end
endmodule
`default_nettype wire
