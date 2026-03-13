module mlp_top (
    input  logic clk,
    input  logic rst,
    input  logic start,
    input  logic signed [7:0] input_vec [0:69],
    output logic done,
    output logic [2:0] class_id
);

    typedef enum logic [2:0] {
        S_IDLE,
        S_FC1_MAC,
        S_FC1_STORE,
        S_FC2_MAC,
        S_FC2_STORE,
        S_ARGMAX,
        S_DONE
    } state_t;

    state_t state;

    logic [6:0] i_idx;
    logic [4:0] h_idx;
    logic [2:0] o_idx;

    logic signed [31:0] acc;
    logic signed [31:0] hidden [0:15];
    logic signed [31:0] outputs [0:4];

    logic signed [7:0]  w1 [0:15][0:69];
    logic signed [31:0] b1 [0:15];
    logic signed [7:0]  w2 [0:4][0:15];
    logic signed [31:0] b2 [0:4];

    logic signed [31:0] max_val;
    logic [2:0] max_idx;

    logic signed [31:0] fc1_product;
    logic signed [31:0] fc2_product;
    logic signed [31:0] acc_next_fc1;
    logic signed [31:0] acc_next_fc2;

    integer k;

`include "mlp_weights.svh"

    // Extend multiply results safely into 32-bit
    always_comb begin
        fc1_product = $signed(input_vec[i_idx]) * $signed(w1[h_idx][i_idx]);
        fc2_product = $signed(hidden[h_idx])    * $signed(w2[o_idx][h_idx]);

        acc_next_fc1 = acc + fc1_product;
        acc_next_fc2 = acc + fc2_product;
    end

    // Combinational argmax
    always_comb begin
        max_val = outputs[0];
        max_idx = 3'd0;

        if (outputs[1] > max_val) begin
            max_val = outputs[1];
            max_idx = 3'd1;
        end
        if (outputs[2] > max_val) begin
            max_val = outputs[2];
            max_idx = 3'd2;
        end
        if (outputs[3] > max_val) begin
            max_val = outputs[3];
            max_idx = 3'd3;
        end
        if (outputs[4] > max_val) begin
            max_val = outputs[4];
            max_idx = 3'd4;
        end
    end

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state    <= S_IDLE;
            done     <= 1'b0;
            class_id <= 3'd0;
            i_idx    <= 7'd0;
            h_idx    <= 5'd0;
            o_idx    <= 3'd0;
            acc      <= 32'sd0;

            for (k = 0; k < 16; k = k + 1)
                hidden[k] <= 32'sd0;

            for (k = 0; k < 5; k = k + 1)
                outputs[k] <= 32'sd0;
        end else begin
            case (state)

                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        h_idx <= 5'd0;
                        i_idx <= 7'd0;
                        acc   <= b1[0];
                        state <= S_FC1_MAC;
                    end
                end

                S_FC1_MAC: begin
                    if (i_idx == 7'd69) begin
                        acc   <= acc_next_fc1;
                        state <= S_FC1_STORE;
                    end else begin
                        acc   <= acc_next_fc1;
                        i_idx <= i_idx + 7'd1;
                    end
                end

                S_FC1_STORE: begin
                    if (acc < 0)
                        hidden[h_idx] <= 32'sd0;
                    else
                        hidden[h_idx] <= acc;

                    if (h_idx == 5'd15) begin
                        o_idx <= 3'd0;
                        h_idx <= 5'd0;
                        acc   <= b2[0];
                        state <= S_FC2_MAC;
                    end else begin
                        h_idx <= h_idx + 5'd1;
                        i_idx <= 7'd0;
                        acc   <= b1[h_idx + 5'd1];
                        state <= S_FC1_MAC;
                    end
                end

                S_FC2_MAC: begin
                    if (h_idx == 5'd15) begin
                        acc   <= acc_next_fc2;
                        state <= S_FC2_STORE;
                    end else begin
                        acc   <= acc_next_fc2;
                        h_idx <= h_idx + 5'd1;
                    end
                end

                S_FC2_STORE: begin
                    outputs[o_idx] <= acc;

                    if (o_idx == 3'd4) begin
                        state <= S_ARGMAX;
                    end else begin
                        o_idx <= o_idx + 3'd1;
                        h_idx <= 5'd0;
                        acc   <= b2[o_idx + 3'd1];
                        state <= S_FC2_MAC;
                    end
                end

                S_ARGMAX: begin
                    class_id <= max_idx;
                    state <= S_DONE;
                end

                S_DONE: begin
                    done <= 1'b1;
                    if (!start)
                        state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule