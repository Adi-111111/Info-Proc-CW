`timescale 1ns/1ps

(* DONT_TOUCH = "yes" *) module rdp_core (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic [7:0]  in_count,
    input  logic [15:0] epsilon,
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

    logic [7:0] stack_lo [0:63];
    logic [7:0] stack_hi [0:63];
    logic [5:0] sp;
    logic keep [0:255];

    typedef enum logic [4:0] {
        IDLE, INIT, POP,
        LOAD_LO, WAIT_LO, LOAD_HI, WAIT_HI,
        SCAN_INIT, LOAD_PT, WAIT_PT,
        CALC_DIST, WAIT_CROSS, WAIT_SQRT_DONE,
        CHECK_DIST, NEXT_PT, DECIDE,
        COLLECT_INIT, COLLECT, DONE_ST
    } state_t;
    state_t state;

    logic signed [FP_WIDTH-1:0] lox, loy, hix, hiy, ptx, pty, abx, aby;
    logic signed [31:0]         ab2;
    logic [7:0]                 lo_idx, hi_idx, scan_idx, max_idx;
    logic signed [FP_WIDTH-1:0] max_dist, dist_val;

    logic signed [31:0] cross_prod;
    logic signed [31:0] cross_prod_abs;
    logic [63:0]        cross_sq;       /
    logic [31:0]        sqrt_radicand;
    logic               sqrt_start, sqrt_done;
    logic [15:0]        sqrt_result;

    logic [7:0] collect_in_idx, collect_out_idx;
    integer i;
    logic [5:0] ab2_shift;
    always_comb begin
        ab2_shift = 6'd0;
        if      (ab2[31]) ab2_shift = 6'd31;
        else if (ab2[30]) ab2_shift = 6'd30;
        else if (ab2[29]) ab2_shift = 6'd29;
        else if (ab2[28]) ab2_shift = 6'd28;
        else if (ab2[27]) ab2_shift = 6'd27;
        else if (ab2[26]) ab2_shift = 6'd26;
        else if (ab2[25]) ab2_shift = 6'd25;
        else if (ab2[24]) ab2_shift = 6'd24;
        else if (ab2[23]) ab2_shift = 6'd23;
        else if (ab2[22]) ab2_shift = 6'd22;
        else if (ab2[21]) ab2_shift = 6'd21;
        else if (ab2[20]) ab2_shift = 6'd20;
        else if (ab2[19]) ab2_shift = 6'd19;
        else if (ab2[18]) ab2_shift = 6'd18;
        else if (ab2[17]) ab2_shift = 6'd17;
        else if (ab2[16]) ab2_shift = 6'd16;
        else if (ab2[15]) ab2_shift = 6'd15;
        else if (ab2[14]) ab2_shift = 6'd14;
        else if (ab2[13]) ab2_shift = 6'd13;
        else if (ab2[12]) ab2_shift = 6'd12;
        else if (ab2[11]) ab2_shift = 6'd11;
        else if (ab2[10]) ab2_shift = 6'd10;
        else if (ab2[ 9]) ab2_shift = 6'd9;
        else if (ab2[ 8]) ab2_shift = 6'd8;
        else if (ab2[ 7]) ab2_shift = 6'd7;
        else if (ab2[ 6]) ab2_shift = 6'd6;
        else if (ab2[ 5]) ab2_shift = 6'd5;
        else if (ab2[ 4]) ab2_shift = 6'd4;
        else if (ab2[ 3]) ab2_shift = 6'd3;
        else if (ab2[ 2]) ab2_shift = 6'd2;
        else if (ab2[ 1]) ab2_shift = 6'd1;
        else              ab2_shift = 6'd0;
    end

    (* DONT_TOUCH = "yes" *) fp_sqrt u_sqrt (
        .clk(clk), .rst_n(rst_n),
        .start(sqrt_start), .radicand(sqrt_radicand),
        .result(sqrt_result), .done(sqrt_done)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; done <= 0; sp <= 0;
            sqrt_start <= 0; out_bram_we <= 0; out_count <= 0;
        end else begin
            sqrt_start  <= 0;
            out_bram_we <= 0;
            done        <= 0;

            case (state)
                IDLE: begin
                    if (start) begin
                        for (i = 0; i < 256; i++) keep[i] <= 1'b0;
                        sp <= 0; state <= INIT;
                    end
                end

                INIT: begin
                    stack_lo[0] <= 8'd0; stack_hi[0] <= in_count - 1;
                    sp <= 6'd1;
                    keep[0] <= 1'b1; keep[in_count-1] <= 1'b1;
                    state <= POP;
                end

                POP: begin
                    if (sp == 0) state <= COLLECT_INIT;
                    else begin
                        sp <= sp - 1;
                        lo_idx <= stack_lo[sp-1]; hi_idx <= stack_hi[sp-1];
                        in_bram_addr <= stack_lo[sp-1];
                        state <= LOAD_LO;
                    end
                end

                LOAD_LO: state <= WAIT_LO;

                WAIT_LO: begin
                    lox <= $signed(in_bram_dout[31:16]);
                    loy <= $signed(in_bram_dout[15:0]);
                    in_bram_addr <= hi_idx; state <= LOAD_HI;
                end

                LOAD_HI: state <= WAIT_HI;

                WAIT_HI: begin
                    hix <= $signed(in_bram_dout[31:16]);
                    hiy <= $signed(in_bram_dout[15:0]);
                    abx <= $signed(in_bram_dout[31:16]) - lox;
                    aby <= $signed(in_bram_dout[15:0])  - loy;
                    ab2 <= ($signed(in_bram_dout[31:16]) - lox) * ($signed(in_bram_dout[31:16]) - lox)
                         + ($signed(in_bram_dout[15:0])  - loy) * ($signed(in_bram_dout[15:0])  - loy);
                    state <= SCAN_INIT;
                end

                SCAN_INIT: begin
                    if (hi_idx <= lo_idx + 1) state <= POP;
                    else begin
                        scan_idx <= lo_idx + 1; max_dist <= '0;
                        max_idx  <= lo_idx + 1;
                        in_bram_addr <= lo_idx + 1; state <= LOAD_PT;
                    end
                end

                LOAD_PT: state <= WAIT_PT;

                WAIT_PT: begin
                    ptx <= $signed(in_bram_dout[31:16]);
                    pty <= $signed(in_bram_dout[15:0]);
                    state <= CALC_DIST;
                end

                CALC_DIST: begin
                    cross_prod <= $signed(abx) * $signed(pty - loy)
                                - $signed(aby) * $signed(ptx - lox);
                    state <= WAIT_CROSS;
                end

                WAIT_CROSS: begin
                    cross_prod_abs <= cross_prod[31] ? -cross_prod : cross_prod;
                    cross_sq       <= (cross_prod[31] ? -cross_prod : cross_prod)
                                    * (cross_prod[31] ? -cross_prod : cross_prod);
                    state <= WAIT_SQRT_DONE; 
                end
                WAIT_SQRT_DONE: begin
                    if (!sqrt_start && !sqrt_done) begin
                        if (ab2 == 0)
                            sqrt_radicand <= (ptx-lox)*(ptx-lox) + (pty-loy)*(pty-loy);
                        else
                            sqrt_radicand <= cross_sq[63:32] >> ab2_shift;
                        sqrt_start <= 1'b1;
                    end else if (sqrt_done) begin
                        dist_val <= $signed({2'b0, sqrt_result[15:2]});
                        state    <= CHECK_DIST;
                    end
                end

                CHECK_DIST: begin
                    if ($signed(dist_val) > $signed(max_dist)) begin
                        max_dist <= dist_val; max_idx <= scan_idx;
                    end
                    state <= NEXT_PT;
                end

                NEXT_PT: begin
                    if (scan_idx >= hi_idx - 1) state <= DECIDE;
                    else begin
                        scan_idx     <= scan_idx + 1;
                        in_bram_addr <= scan_idx + 1; state <= LOAD_PT;
                    end
                end

                DECIDE: begin
                    if ($signed(max_dist) > $signed(epsilon)) begin
                        if (sp < 62) begin
                            keep[max_idx]      <= 1'b1;
                            stack_lo[sp]       <= lo_idx;
                            stack_hi[sp]       <= max_idx;
                            stack_lo[sp+1]     <= max_idx;
                            stack_hi[sp+1]     <= hi_idx;
                            sp <= sp + 2;
                        end
                    end
                    state <= POP;
                end

                COLLECT_INIT: begin
                    collect_in_idx  <= 8'd0;
                    collect_out_idx <= 8'd0;
                    state <= COLLECT;
                end

                COLLECT: begin
                    if (collect_in_idx >= in_count) begin
                        out_count <= collect_out_idx; state <= DONE_ST;
                    end else begin
                        if (keep[collect_in_idx]) in_bram_addr <= collect_in_idx;
                        collect_in_idx <= collect_in_idx + 1;
                    end
                    if (collect_in_idx > 0 && keep[collect_in_idx-1]) begin
                        out_bram_addr   <= collect_out_idx;
                        out_bram_din    <= in_bram_dout;
                        out_bram_we     <= 1'b1;
                        collect_out_idx <= collect_out_idx + 1;
                    end
                end

                DONE_ST: begin done <= 1'b1; state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
