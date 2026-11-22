// ternary_dot16: multiplier-free dot product of 16 ternary weights with 16 int8 activations.
//
// Weights arrive as one 32-bit word in the crate's *natural* packed layout: weight t sits in
// bits [2t+1:2t] as code (w + 1), i.e. 0 -> -1, 1 -> 0, 2 -> +1 (3 is invalid and contributes 0,
// matching PackedTernary::from_natural, which rejects it).
// Activations arrive as 16 bytes, activation t in bits [8t+7:8t] (two's complement).
//
// Each lane is a 2:1 mux + negate (no multipliers); a balanced adder tree sums the lanes.
// |sum| <= 16 * 128 = 2048, so 13 bits are enough; the output is sign-extended to 32 bits.
// One register stage: result is valid one cycle after in_valid.
`timescale 1ns/1ps
`default_nettype none
module ternary_dot16 (
    input  wire         clk,
    input  wire         rst,
    input  wire         in_valid,
    input  wire [31:0]  w_codes,
    input  wire [127:0] acts,
    output reg          out_valid,
    output reg  signed [31:0] dot
);
    // Lane products in {-a, 0, +a}, 9 bits signed (covers -(-128) = +128).
    wire signed [8:0] lane [0:15];
    genvar t;
    generate
        for (t = 0; t < 16; t = t + 1) begin : g_lane
            wire [1:0]        c = w_codes[2*t+1:2*t];
            wire signed [8:0] a = {acts[8*t+7], acts[8*t+7:8*t]};
            assign lane[t] = (c == 2'd2) ? a : (c == 2'd0) ? -a : 9'sd0;
        end
    endgenerate

    // Adder tree: 16 -> 8 -> 4 -> 2 -> 1.
    wire signed [9:0]  s1 [0:7];
    wire signed [10:0] s2 [0:3];
    wire signed [11:0] s3 [0:1];
    wire signed [12:0] s4;
    generate
        for (t = 0; t < 8; t = t + 1) begin : g_s1
            assign s1[t] = lane[2*t] + lane[2*t+1];
        end
        for (t = 0; t < 4; t = t + 1) begin : g_s2
            assign s2[t] = s1[2*t] + s1[2*t+1];
        end
        for (t = 0; t < 2; t = t + 1) begin : g_s3
            assign s3[t] = s2[2*t] + s2[2*t+1];
        end
    endgenerate
    assign s4 = s3[0] + s3[1];

    always @(posedge clk) begin
        if (rst) begin
            out_valid <= 1'b0;
            dot       <= 32'sd0;
        end else begin
            out_valid <= in_valid;
            if (in_valid) dot <= {{19{s4[12]}}, s4};
        end
    end
endmodule
`default_nettype wire
