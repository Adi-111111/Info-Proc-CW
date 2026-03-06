// resample_core.sv

`timescale 1ns/1ps

(* DONT_TOUCH = "yes" *) module resample_core (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic [7:0]  in_count,
    input  logic [7:0]  step_fp,
    output logic [7:0]  in_bram_addr,
    input  logic [31:0] in_bram_dout,
    output logic [7:0]  out_bram_addr,
    output logic [31:0] out_bram_din,
    output logic        out_bram_we,
    output logic [7:0]  out_count,
    output logic        done
);
    localparam integer FP_WIDTH = 16;
    localparam integer FP_FRAC  = 6;

    typedef enum logic [3:0] {
        IDLE, LOAD_FIRST, WAIT_FIRST, EMIT_FIRST,
        LOAD_NEXT, WAIT_NEXT, COMPUTE_SEG, WAIT_SQRT,
        CALC_T, EMIT_PT, ADVANCE, EMIT_LAST, DONE_ST
    } state_t;
    state_t state;

    // All registers at module level
    logic signed [15:0] px, py, cx, cy, acc, seg, dx, dy;
    logic signed [15:0] t_reg;  
    logic signed [15:0] newx, newy;
    logic [7:0] in_idx, out_idx;

    logic        sqrt_start, sqrt_done;
    logic [31:0] sqrt_radicand;
    logic [15:0] sqrt_result;

    logic signed [15:0] step_val;
    assign step_val = {8'b0, step_fp};

    logic signed [31:0] t_dx, t_dy;

    // Leading-zero shift for seg
    logic [3:0] seg_shift;
    always_comb begin
        seg_shift = 4'd0;
        if      (seg[15]) seg_shift = 4'd15;
        else if (seg[14]) seg_shift = 4'd14;
        else if (seg[13]) seg_shift = 4'd13;
        else if (seg[12]) seg_shift = 4'd12;
        else if (seg[11]) seg_shift = 4'd11;
        else if (seg[10]) seg_shift = 4'd10;
        else if (seg[ 9]) seg_shift = 4'd9;
        else if (seg[ 8]) seg_shift = 4'd8;
        else if (seg[ 7]) seg_shift = 4'd7;
        else if (seg[ 6]) seg_shift = 4'd6;
        else if (seg[ 5]) seg_shift = 4'd5;
        else if (seg[ 4]) seg_shift = 4'd4;
        else if (seg[ 3]) seg_shift = 4'd3;
        else if (seg[ 2]) seg_shift = 4'd2;
        else if (seg[ 1]) seg_shift = 4'd1;
        else              seg_shift = 4'd0;
    end

    (* DONT_TOUCH = "yes" *) fp_sqrt u_sqrt (
        .clk(clk), .rst_n(rst_n),
        .start(sqrt_start), .radicand(sqrt_radicand),
        .result(sqrt_result), .done(sqrt_done)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; done <= 0; out_count <= 0;
            in_bram_addr <= 0; out_bram_addr <= 0;
            out_bram_we <= 0; sqrt_start <= 0;
            acc <= 0; in_idx <= 0; out_idx <= 0;
        end else begin
            sqrt_start  <= 0;
            out_bram_we <= 0;
            done        <= 0;

            case (state)
                IDLE: begin
                    if (start) begin
                        in_idx <= 0; out_idx <= 0; acc <= 0;
                        in_bram_addr <= 0; state <= LOAD_FIRST;
                    end
                end

                LOAD_FIRST: state <= WAIT_FIRST;

                WAIT_FIRST: begin
                    px <= $signed(in_bram_dout[31:16]);
                    py <= $signed(in_bram_dout[15:0]);
                    state <= EMIT_FIRST;
                end

                EMIT_FIRST: begin
                    out_bram_addr <= 0; out_bram_din <= in_bram_dout;
                    out_bram_we <= 1; out_idx <= 1; in_idx <= 1;
                    in_bram_addr <= 1; state <= LOAD_NEXT;
                end

                LOAD_NEXT: begin
                    if (in_idx >= in_count) state <= EMIT_LAST;
                    else state <= WAIT_NEXT;
                end

                WAIT_NEXT: begin
                    cx <= $signed(in_bram_dout[31:16]);
                    cy <= $signed(in_bram_dout[15:0]);
                    state <= COMPUTE_SEG;
                end

                COMPUTE_SEG: begin
                    dx <= $signed(cx) - $signed(px);
                    dy <= $signed(cy) - $signed(py);
                    sqrt_radicand <= ($signed(cx-px)*$signed(cx-px)
                                    + $signed(cy-py)*$signed(cy-py));
                    sqrt_start <= 1; state <= WAIT_SQRT;
                end

                WAIT_SQRT: begin
                    if (sqrt_done) begin
                        seg <= $signed({2'b0, sqrt_result[15:2]});
                        state <= ADVANCE; 
                    end
                end

                // After getting seg: decide whether to emit a point
                ADVANCE: begin
                    if (seg >= 16'sd1 && (acc + seg >= step_val) && out_idx < 8'd255) begin
                        t_reg <= ($signed(step_val - acc) <<< 6) >>> seg_shift;
                        state <= CALC_T;
                    end else begin
                        acc <= acc + seg; px <= cx; py <= cy;
                        in_idx <= in_idx + 1; in_bram_addr <= in_idx + 1;
                        state <= LOAD_NEXT;
                    end
                end

                CALC_T: begin
                    t_dx  <= $signed(t_reg) * $signed(dx);
                    t_dy  <= $signed(t_reg) * $signed(dy);
                    state <= EMIT_PT;
                end

                EMIT_PT: begin
                    newx <= $signed(px) + $signed(t_dx >>> 6);
                    newy <= $signed(py) + $signed(t_dy >>> 6);
                    out_bram_addr <= out_idx;
                    out_bram_din  <= {$signed(px) + $signed(t_dx>>>6),
                                      $signed(py) + $signed(t_dy>>>6)};
                    out_bram_we  <= 1;
                    out_idx      <= out_idx + 1;
                    px  <= $signed(px) + $signed(t_dx >>> 6);
                    py  <= $signed(py) + $signed(t_dy >>> 6);
                    acc <= 0;
                    state <= COMPUTE_SEG;
                end

                EMIT_LAST: begin
                    in_bram_addr <= in_count - 1; state <= DONE_ST;
                end

                DONE_ST: begin
                    out_count <= out_idx; done <= 1; state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end
endmodule
