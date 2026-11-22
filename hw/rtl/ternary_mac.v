// ternary_mac: accumulates ternary_dot16 results over the K/16 words of one output row.
//
// Stream one (weight word, 16 activations) pair per cycle with in_valid. Assert `first` on the
// first word of a row (clears the accumulator) and `last` on the final word; `acc_valid` pulses
// with the finished int32 row sum two cycles after the `last` beat. This is the integer part of
// the W2A8 GEMV (`acc = sum_k w[k] * q[k]`); the f32 epilogue (`acc / s * scale`) stays on the
// host, exactly as in the crate's reference.
`timescale 1ns/1ps
`default_nettype none
module ternary_mac (
    input  wire         clk,
    input  wire         rst,
    input  wire         in_valid,
    input  wire         first,
    input  wire         last,
    input  wire [31:0]  w_codes,
    input  wire [127:0] acts,
    output reg          acc_valid,
    output reg  signed [31:0] acc
);
    wire               dot_valid;
    wire signed [31:0] dot;
    reg                first_d, last_d;
    reg  signed [31:0] running;

    ternary_dot16 u_dot (
        .clk(clk), .rst(rst), .in_valid(in_valid),
        .w_codes(w_codes), .acts(acts),
        .out_valid(dot_valid), .dot(dot)
    );

    always @(posedge clk) begin
        if (rst) begin
            first_d   <= 1'b0;
            last_d    <= 1'b0;
            running   <= 32'sd0;
            acc_valid <= 1'b0;
            acc       <= 32'sd0;
        end else begin
            first_d   <= first & in_valid;
            last_d    <= last & in_valid;
            acc_valid <= 1'b0;
            if (dot_valid) begin
                if (first_d) running <= dot;
                else         running <= running + dot;
                if (last_d) begin
                    acc       <= first_d ? dot : running + dot;
                    acc_valid <= 1'b1;
                end
            end
        end
    end
endmodule
`default_nettype wire
