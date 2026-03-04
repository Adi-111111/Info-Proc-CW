`timescale 1ns/1ps

(* KEEP_HIERARCHY = "yes" *) (* DONT_TOUCH = "yes" *) module fp_sqrt (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic [31:0] radicand,
    output logic [15:0] result,
    output logic        done
);
    logic [31:0] rem;
    logic [15:0] root;
    logic [4:0]  iter;
    logic        busy;

    logic [31:0] trial_val;
    assign trial_val = rem - ({root, 2'b01} << ((iter - 1) * 2));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rem    <= '0;
            root   <= '0;
            iter   <= '0;
            busy   <= 1'b0;
            done   <= 1'b0;
            result <= '0;
        end else if (start && !busy) begin
            rem    <= radicand;
            root   <= '0;
            iter   <= 5'd16;
            busy   <= 1'b1;
            done   <= 1'b0;
        end else if (busy) begin
            if (iter > 0) begin
                if (!trial_val[31]) begin
                    root <= (root << 1) | 1'b1;
                    rem  <= trial_val;
                end else begin
                    root <= root << 1;
                end
                iter <= iter - 1;
                done <= 1'b0;
            end else begin
                result <= root;
                done   <= 1'b1;
                busy   <= 1'b0;
            end
        end else begin
            done <= 1'b0;
        end
    end
endmodule
