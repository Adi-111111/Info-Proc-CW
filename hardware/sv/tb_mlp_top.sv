module tb_mlp_top;

    logic clk;
    logic rst;
    logic start;
    logic signed [7:0] input_vec [0:69];
    logic done;
    logic [2:0] class_id;

    mlp_top dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .input_vec(input_vec),
        .done(done),
        .class_id(class_id)
    );

    always #5 clk = ~clk;

`include "test_vector.svh"

    initial begin
        clk = 0;
        rst = 1;
        start = 0;

        #20;
        rst = 0;

        #20;
        start = 1;
        #10;
        start = 0;

        wait(done == 1);

        $display("DONE. class_id = %0d", class_id);
        #20;
        $finish;
    end

endmodule