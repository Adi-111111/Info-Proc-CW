`timescale 1ns/1ps

module kasa_core (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic [7:0]  in_count,
    output logic [7:0]  in_bram_addr,
    input  logic [31:0] in_bram_dout,
    output logic signed [FP_WIDTH-1:0] cx_out,
    output logic signed [FP_WIDTH-1:0] cy_out,
    output logic signed [FP_WIDTH-1:0] r_out,
    output logic        valid,
    output logic        done
);
    localparam integer FP_WIDTH = 16;
    localparam integer FP_FRAC  = 6;

    typedef enum logic [3:0] {
        IDLE, LOAD_PT, WAIT_PT, ACCUMULATE, NEXT_PT,
        SOLVE, SQRT_CX2, WAIT_SQRT_CX2, SQRT_R, WAIT_SQRT_R, DONE_ST
    } state_t;
    state_t state;

    logic signed [47:0] sum_x, sum_y, sum_x2, sum_y2, sum_xy;
    logic signed [47:0] sum_xr, sum_yr, sum_r2;
    logic [7:0] pt_idx;
    logic signed [FP_WIDTH-1:0] ptx, pty;
    logic signed [2*FP_WIDTH-1:0] ptx2, pty2, ptxy, ptr2;
    logic signed [47:0] a_sol, b_sol, c_sol;
    logic signed [FP_WIDTH-1:0] cx_reg, cy_reg;
    logic signed [47:0] n_fp;
    logic signed [47:0] det, det_a, det_b, det_c;
    logic signed [47:0] r2;

    logic [5:0] det_shift;
    always_comb begin
        det_shift = 6'd0;
        if      (det[46]) det_shift = 6'd46;
        else if (det[45]) det_shift = 6'd45;
        else if (det[44]) det_shift = 6'd44;
        else if (det[43]) det_shift = 6'd43;
        else if (det[42]) det_shift = 6'd42;
        else if (det[41]) det_shift = 6'd41;
        else if (det[40]) det_shift = 6'd40;
        else if (det[39]) det_shift = 6'd39;
        else if (det[38]) det_shift = 6'd38;
        else if (det[37]) det_shift = 6'd37;
        else if (det[36]) det_shift = 6'd36;
        else if (det[35]) det_shift = 6'd35;
        else if (det[34]) det_shift = 6'd34;
        else if (det[33]) det_shift = 6'd33;
        else if (det[32]) det_shift = 6'd32;
        else if (det[31]) det_shift = 6'd31;
        else if (det[30]) det_shift = 6'd30;
        else if (det[29]) det_shift = 6'd29;
        else if (det[28]) det_shift = 6'd28;
        else if (det[27]) det_shift = 6'd27;
        else if (det[26]) det_shift = 6'd26;
        else if (det[25]) det_shift = 6'd25;
        else if (det[24]) det_shift = 6'd24;
        else if (det[23]) det_shift = 6'd23;
        else if (det[22]) det_shift = 6'd22;
        else if (det[21]) det_shift = 6'd21;
        else if (det[20]) det_shift = 6'd20;
        else if (det[19]) det_shift = 6'd19;
        else if (det[18]) det_shift = 6'd18;
        else if (det[17]) det_shift = 6'd17;
        else if (det[16]) det_shift = 6'd16;
        else              det_shift = 6'd15;
    end

    logic        sqrt_start, sqrt_done;
    logic [31:0] sqrt_radicand;
    logic [15:0] sqrt_result;

    fp_sqrt u_sqrt (
        .clk(clk), .rst_n(rst_n),
        .start(sqrt_start), .radicand(sqrt_radicand),
        .result(sqrt_result), .done(sqrt_done)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; done <= 0; valid <= 0; sqrt_start <= 0;
            cx_out <= 0; cy_out <= 0; r_out <= 0;
        end else begin
            sqrt_start <= 0;
            done <= 0;

            case (state)
                IDLE: begin
                    if (start) begin
                        sum_x <= 0; sum_y <= 0; sum_x2 <= 0; sum_y2 <= 0;
                        sum_xy <= 0; sum_xr <= 0; sum_yr <= 0; sum_r2 <= 0;
                        pt_idx <= 0; valid <= 0;
                        in_bram_addr <= 0; state <= LOAD_PT;
                    end
                end

                LOAD_PT: state <= WAIT_PT;

                WAIT_PT: begin
                    ptx <= $signed(in_bram_dout[31:16]);
                    pty <= $signed(in_bram_dout[15:0]);
                    state <= ACCUMULATE;
                end

                ACCUMULATE: begin
                    ptx2 <= $signed(ptx) * $signed(ptx);
                    pty2 <= $signed(pty) * $signed(pty);
                    ptxy <= $signed(ptx) * $signed(pty);
                    ptr2 <= $signed(ptx)*$signed(ptx) + $signed(pty)*$signed(pty);
                    sum_x  <= sum_x  + ptx;
                    sum_y  <= sum_y  + pty;
                    sum_x2 <= sum_x2 + $signed(ptx) * $signed(ptx);
                    sum_y2 <= sum_y2 + $signed(pty) * $signed(pty);
                    sum_xy <= sum_xy + $signed(ptx) * $signed(pty);
                    sum_xr <= sum_xr + $signed(ptx) * ($signed(ptx)*$signed(ptx) + $signed(pty)*$signed(pty));
                    sum_yr <= sum_yr + $signed(pty) * ($signed(ptx)*$signed(ptx) + $signed(pty)*$signed(pty));
                    sum_r2 <= sum_r2 + $signed(ptx)*$signed(ptx) + $signed(pty)*$signed(pty);
                    state <= NEXT_PT;
                end

                NEXT_PT: begin
                    if (pt_idx >= in_count - 1) state <= SOLVE;
                    else begin
                        pt_idx <= pt_idx + 1;
                        in_bram_addr <= pt_idx + 1;
                        state <= LOAD_PT;
                    end
                end

                SOLVE: begin
                    n_fp <= {{40{1'b0}}, in_count} <<< FP_FRAC;
                    det <= (sum_x2 >>> FP_FRAC) * ((sum_y2 * n_fp - sum_y * sum_y) >>> FP_FRAC)
                         - (sum_xy >>> FP_FRAC) * ((sum_xy * n_fp - sum_y * sum_x) >>> FP_FRAC)
                         + (sum_x  >>> FP_FRAC) * ((sum_xy * sum_y - sum_y2 * sum_x) >>> FP_FRAC);

                    det_a <= (sum_xr >>> FP_FRAC) * ((sum_y2 * n_fp - sum_y * sum_y) >>> FP_FRAC)
                           - (sum_xy >>> FP_FRAC) * ((sum_yr * n_fp - sum_y * sum_r2) >>> FP_FRAC)
                           + (sum_x  >>> FP_FRAC) * ((sum_yr * sum_y - sum_y2 * sum_r2) >>> FP_FRAC);

                    det_b <= (sum_x2 >>> FP_FRAC) * ((sum_yr * n_fp - sum_r2 * sum_y) >>> FP_FRAC)
                           - (sum_xr >>> FP_FRAC) * ((sum_xy * n_fp - sum_y * sum_x) >>> FP_FRAC)
                           + (sum_x  >>> FP_FRAC) * ((sum_xy * sum_r2 - sum_yr * sum_x) >>> FP_FRAC);

                    det_c <= (sum_x2 >>> FP_FRAC) * ((sum_y2 * sum_r2 - sum_yr * sum_y) >>> FP_FRAC)
                           - (sum_xy >>> FP_FRAC) * ((sum_xy * sum_r2 - sum_yr * sum_x) >>> FP_FRAC)
                           + (sum_xr >>> FP_FRAC) * ((sum_xy * sum_y - sum_y2 * sum_x) >>> FP_FRAC);

                    state <= SQRT_CX2; 
                end

                SQRT_CX2: begin
                    if (det == 0) begin
                        valid <= 0; done <= 1; state <= IDLE;
                    end else begin
                        a_sol <= det_a >>> det_shift;
                        b_sol <= det_b >>> det_shift;
                        c_sol <= det_c >>> det_shift;
                        state <= WAIT_SQRT_CX2;
                    end
                end

                WAIT_SQRT_CX2: begin
                    cx_reg <= $signed(a_sol[FP_WIDTH:1]);
                    cy_reg <= $signed(b_sol[FP_WIDTH:1]);
                    sqrt_radicand <= (a_sol[FP_WIDTH:1] * a_sol[FP_WIDTH:1]
                                    + b_sol[FP_WIDTH:1] * b_sol[FP_WIDTH:1]) >>> FP_FRAC;
                    sqrt_start <= 1;
                    state <= SQRT_R;
                end

                SQRT_R: begin
                    if (sqrt_done) begin
                        r2 <= {32'b0, sqrt_result} * {32'b0, sqrt_result} + (c_sol <<< FP_FRAC);
                        state <= WAIT_SQRT_R;
                    end
                end

                WAIT_SQRT_R: begin
                    if (r2 <= 0) begin
                        valid <= 0; done <= 1; state <= IDLE;
                    end else begin
                        sqrt_radicand <= r2[31:0];
                        sqrt_start <= 1;
                        state <= DONE_ST;
                    end
                end

                DONE_ST: begin
                    if (sqrt_done) begin
                        cx_out <= cx_reg;
                        cy_out <= cy_reg;
                        r_out  <= $signed({2'b0, sqrt_result[15:2]});
                        valid  <= 1; done <= 1; state <= IDLE;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end
endmodule
