`timescale 1ns/1ps

module rect_detect_core (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic signed [FP_WIDTH-1:0] x0, y0,
    input  logic signed [FP_WIDTH-1:0] x1, y1,
    input  logic signed [FP_WIDTH-1:0] x2, y2,
    input  logic signed [FP_WIDTH-1:0] x3, y3,
    input  logic signed [FP_WIDTH-1:0] angle_tol,

    output logic        done,
    output logic        is_rect,
    output logic signed [FP_WIDTH-1:0] max_angle_err 
);
    localparam integer FP_WIDTH = 16;
    localparam integer FP_FRAC  = 6;

    localparam signed [FP_WIDTH-1:0] COS_TOL = 16'sd41;
    typedef enum logic [2:0] {
        IDLE, COMPUTE_0, COMPUTE_1, COMPUTE_2, COMPUTE_3, DONE_ST
    } state_t;
    state_t state;
    logic signed [FP_WIDTH-1:0] cx [0:3];
    logic signed [FP_WIDTH-1:0] cy [0:3];

    logic signed [FP_WIDTH-1:0] v1x, v1y, v2x, v2y;
    logic signed [2*FP_WIDTH-1:0] dot, mag1_sq, mag2_sq;
    logic signed [2*FP_WIDTH-1:0] dot_sq, mag_sq;

    logic [1:0] corner_idx;
    logic [3:0] angle_ok;

    logic signed [FP_WIDTH-1:0] angle_err_cur;
    logic signed [FP_WIDTH-1:0] max_err_reg;

    function automatic [1:0] prev_idx;
        input [1:0] i;
        prev_idx = (i == 2'd0) ? 2'd3 : i - 2'd1;
    endfunction

    function automatic [1:0] next_idx;
        input [1:0] i;
        next_idx = (i == 2'd3) ? 2'd0 : i + 2'd1;
    endfunction

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= IDLE;
            done        <= 1'b0;
            is_rect     <= 1'b0;
            max_err_reg <= '0;
            corner_idx  <= '0;
            angle_ok    <= 4'b1111;
        end else begin
            case (state)
                IDLE: begin
                    done    <= 1'b0;
                    is_rect <= 1'b0;
                    angle_ok <= 4'b1111;
                    max_err_reg <= '0;
                    if (start) begin
                        cx[0] <= x0; cy[0] <= y0;
                        cx[1] <= x1; cy[1] <= y1;
                        cx[2] <= x2; cy[2] <= y2;
                        cx[3] <= x3; cy[3] <= y3;
                        corner_idx <= 2'd0;
                        state <= COMPUTE_0;
                    end
                end

                COMPUTE_0, COMPUTE_1, COMPUTE_2, COMPUTE_3: begin
                    v1x = cx[prev_idx(corner_idx)] - cx[corner_idx];
                    v1y = cy[prev_idx(corner_idx)] - cy[corner_idx];
                    v2x = cx[next_idx(corner_idx)] - cx[corner_idx];
                    v2y = cy[next_idx(corner_idx)] - cy[corner_idx];
                    dot     = $signed(v1x) * $signed(v2x) + $signed(v1y) * $signed(v2y);
                    mag1_sq = $signed(v1x) * $signed(v1x) + $signed(v1y) * $signed(v1y);
                    mag2_sq = $signed(v2x) * $signed(v2x) + $signed(v2y) * $signed(v2y);
                    mag_sq  = (mag1_sq >>> FP_FRAC) * (mag2_sq >>> FP_FRAC);

                    dot_sq  = dot * dot;
                    if (mag_sq > 0) begin
                        angle_ok[corner_idx] <= (dot_sq <= (26 * mag_sq));
                    end else begin
                        angle_ok[corner_idx] <= 1'b0;
                    end

                    if (corner_idx == 2'd3) begin
                        state <= DONE_ST;
                    end else begin
                        corner_idx <= corner_idx + 1;
                        case (corner_idx)
                            2'd0: state <= COMPUTE_1;
                            2'd1: state <= COMPUTE_2;
                            2'd2: state <= COMPUTE_3;
                            default: state <= DONE_ST;
                        endcase
                    end
                end

                DONE_ST: begin
                    is_rect     <= &angle_ok;
                    max_err_reg <= '0;
                    done        <= 1'b1;
                    state       <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

    assign max_angle_err = max_err_reg;

endmodule
