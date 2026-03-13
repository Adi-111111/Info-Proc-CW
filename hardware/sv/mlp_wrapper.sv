module mlp_wrapper (
    input  logic clk,
    input  logic rst,
    input  logic start,
    input  logic signed [70*8-1:0] input_bus,
    output logic done,
    output logic [2:0] class_id
);

    logic signed [7:0] input_vec [0:69];

    genvar i;
    generate
        for (i = 0; i < 70; i = i + 1) begin : UNPACK_INPUTS
            assign input_vec[i] = input_bus[i*8 +: 8];
        end
    endgenerate

    mlp_top u_mlp_top (
        .clk(clk),
        .rst(rst),
        .start(start),
        .input_vec(input_vec),
        .done(done),
        .class_id(class_id)
    );

endmodule
