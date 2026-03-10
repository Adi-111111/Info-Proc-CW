module tb_mlp_wrapper;

    logic clk;
    logic rst;
    logic start;
    logic signed [70*8-1:0] input_bus;
    logic done;
    logic [2:0] class_id;

    mlp_wrapper dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .input_bus(input_bus),
        .done(done),
        .class_id(class_id)
    );

    always #5 clk = ~clk;

    initial begin
        clk = 0;
        rst = 1;
        start = 0;
        input_bus = '0;

        #20;
        rst = 0;

        // Load real test vector into input_bus
`include "test_vector_wrapper.svh"

        #20;
        start = 1;
        #10;
        start = 0;

        $display("Simulation started, waiting for done...");
        wait(done == 1);

        $display("DONE at time %0t", $time);
        $display("Predicted class_id = %0d", class_id);

        #20;
        $finish;
    end

endmodule