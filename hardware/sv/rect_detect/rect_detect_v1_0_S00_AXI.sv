// rect_detect_v1_0.sv
// -------------------
// AXI4-Lite wrapper for rectangle detection IP.
// Simpler than the others — no BRAM, just 11 scalar registers.
//
// Register map:
//   slv_reg0   0x00  CTRL:       [0]=start, [1]=done, [2]=is_rect (all read after start)
//   slv_reg1   0x04  X0:         corner 0 x Q10.6
//   slv_reg2   0x08  Y0:         corner 0 y Q10.6
//   slv_reg3   0x0C  X1:         corner 1 x Q10.6
//   slv_reg4   0x10  Y1:         corner 1 y Q10.6
//   slv_reg5   0x14  X2:         corner 2 x Q10.6
//   slv_reg6   0x18  Y2:         corner 2 y Q10.6
//   slv_reg7   0x1C  X3:         corner 3 x Q10.6
//   slv_reg8   0x20  Y3:         corner 3 y Q10.6
//   slv_reg9   0x24  ANGLE_TOL:  tolerance Q10.6 (50.0 = 3200)
//   slv_reg10  0x28  MAX_ERR:    max angle error output Q10.6 (read only)
//
// Note: 11 registers require OPT_MEM_ADDR_BITS=3 (addr bits [4:2])

`timescale 1ns/1ps

module rect_detect_v1_0_S00_AXI #(
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 6
)(
    input  logic                               S_AXI_ACLK,
    input  logic                               S_AXI_ARESETN,
    input  logic [C_S_AXI_ADDR_WIDTH-1:0]     S_AXI_AWADDR,
    input  logic [2:0]                         S_AXI_AWPROT,
    input  logic                               S_AXI_AWVALID,
    output logic                               S_AXI_AWREADY,
    input  logic [C_S_AXI_DATA_WIDTH-1:0]     S_AXI_WDATA,
    input  logic [(C_S_AXI_DATA_WIDTH/8)-1:0] S_AXI_WSTRB,
    input  logic                               S_AXI_WVALID,
    output logic                               S_AXI_WREADY,
    output logic [1:0]                         S_AXI_BRESP,
    output logic                               S_AXI_BVALID,
    input  logic                               S_AXI_BREADY,
    input  logic [C_S_AXI_ADDR_WIDTH-1:0]     S_AXI_ARADDR,
    input  logic [2:0]                         S_AXI_ARPROT,
    input  logic                               S_AXI_ARVALID,
    output logic                               S_AXI_ARREADY,
    output logic [C_S_AXI_DATA_WIDTH-1:0]     S_AXI_RDATA,
    output logic [1:0]                         S_AXI_RRESP,
    output logic                               S_AXI_RVALID,
    input  logic                               S_AXI_RREADY
);
    localparam integer FP_WIDTH = 16;
    localparam integer FP_FRAC  = 6;
    localparam integer ADDR_LSB = 2, OPT_MEM_ADDR_BITS = 3;

    logic [C_S_AXI_ADDR_WIDTH-1:0] axi_awaddr, axi_araddr;
    logic axi_awready,axi_wready,axi_bvalid,axi_arready,axi_rvalid,aw_en;
    logic [1:0] axi_bresp, axi_rresp;
    logic [C_S_AXI_DATA_WIDTH-1:0] axi_rdata, reg_data_out;
    logic [C_S_AXI_DATA_WIDTH-1:0] slv_reg0,slv_reg1,slv_reg2,slv_reg3,slv_reg4;
    logic [C_S_AXI_DATA_WIDTH-1:0] slv_reg5,slv_reg6,slv_reg7,slv_reg8,slv_reg9;
    logic slv_reg_rden, slv_reg_wren;

    assign S_AXI_AWREADY=axi_awready; assign S_AXI_WREADY=axi_wready;
    assign S_AXI_BRESP=axi_bresp;     assign S_AXI_BVALID=axi_bvalid;
    assign S_AXI_ARREADY=axi_arready; assign S_AXI_RDATA=axi_rdata;
    assign S_AXI_RRESP=axi_rresp;     assign S_AXI_RVALID=axi_rvalid;

    logic        core_done, core_is_rect;
    logic        done_latch;
    logic signed [FP_WIDTH-1:0] core_max_err;

    rect_detect_core u_rect (
        .clk       (S_AXI_ACLK),
        .rst_n     (S_AXI_ARESETN),
        .start     (slv_reg0[0]),
        .x0        (slv_reg1[15:0]), .y0(slv_reg2[15:0]),
        .x1        (slv_reg3[15:0]), .y1(slv_reg4[15:0]),
        .x2        (slv_reg5[15:0]), .y2(slv_reg6[15:0]),
        .x3        (slv_reg7[15:0]), .y3(slv_reg8[15:0]),
        .angle_tol (slv_reg9[15:0]),
        .done      (core_done),
        .is_rect   (core_is_rect),
        .max_angle_err(core_max_err)
    );

    always_ff @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin axi_awready<=0; aw_en<=1; end
        else begin
            if (!axi_awready&&S_AXI_AWVALID&&S_AXI_WVALID&&aw_en)
                begin axi_awready<=1; axi_awaddr<=S_AXI_AWADDR; aw_en<=0; end
            else if (S_AXI_BREADY&&axi_bvalid) begin aw_en<=1; axi_awready<=0; end
            else axi_awready<=0;
        end
    end
    always_ff @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) axi_wready<=0;
        else axi_wready<=(!axi_wready&&S_AXI_WVALID&&S_AXI_AWVALID&&aw_en);
    end
    assign slv_reg_wren=axi_wready&&S_AXI_WVALID&&axi_awready&&S_AXI_AWVALID;

    always_ff @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin slv_reg0<=0; done_latch<=0; end
        else begin
            if (core_done) begin slv_reg0<=0; done_latch<=1'b1; end
            if (slv_reg_wren&&axi_awaddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB]==4'h0)
                begin slv_reg0<=S_AXI_WDATA; done_latch<=1'b0; end
        end
    end
    always_ff @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            slv_reg1<=0; slv_reg2<=0; slv_reg3<=0; slv_reg4<=0;
            slv_reg5<=0; slv_reg6<=0; slv_reg7<=0; slv_reg8<=0;
            slv_reg9<=32'd3200; // default 50.0*64
        end else if (slv_reg_wren) begin
            case (axi_awaddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB])
                4'h1: slv_reg1<=S_AXI_WDATA;
                4'h2: slv_reg2<=S_AXI_WDATA;
                4'h3: slv_reg3<=S_AXI_WDATA;
                4'h4: slv_reg4<=S_AXI_WDATA;
                4'h5: slv_reg5<=S_AXI_WDATA;
                4'h6: slv_reg6<=S_AXI_WDATA;
                4'h7: slv_reg7<=S_AXI_WDATA;
                4'h8: slv_reg8<=S_AXI_WDATA;
                4'h9: slv_reg9<=S_AXI_WDATA;
                default:;
            endcase
        end
    end

    always_ff @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin axi_bvalid<=0; axi_bresp<=0; end
        else begin
            if (axi_awready&&S_AXI_AWVALID&&!axi_bvalid&&axi_wready&&S_AXI_WVALID)
                begin axi_bvalid<=1; axi_bresp<=0; end
            else if (S_AXI_BREADY&&axi_bvalid) axi_bvalid<=0;
        end
    end
    always_ff @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin axi_arready<=0; axi_araddr<=0; end
        else begin
            if (!axi_arready&&S_AXI_ARVALID) begin axi_arready<=1; axi_araddr<=S_AXI_ARADDR; end
            else axi_arready<=0;
        end
    end
    always_ff @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin axi_rvalid<=0; axi_rresp<=0; end
        else begin
            if (axi_arready&&S_AXI_ARVALID&&!axi_rvalid) begin axi_rvalid<=1; axi_rresp<=0; end
            else if (axi_rvalid&&S_AXI_RREADY) axi_rvalid<=0;
        end
    end
    assign slv_reg_rden=axi_arready&S_AXI_ARVALID&~axi_rvalid;

    always_comb begin
        case (axi_araddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB])
            4'h0: reg_data_out = {29'b0, core_is_rect, done_latch, slv_reg0[0]};
            4'h1: reg_data_out = slv_reg1;
            4'h2: reg_data_out = slv_reg2;
            4'h3: reg_data_out = slv_reg3;
            4'h4: reg_data_out = slv_reg4;
            4'h5: reg_data_out = slv_reg5;
            4'h6: reg_data_out = slv_reg6;
            4'h7: reg_data_out = slv_reg7;
            4'h8: reg_data_out = slv_reg8;
            4'h9: reg_data_out = slv_reg9;
            4'hA: reg_data_out = {16'b0, core_max_err};
            default: reg_data_out = 0;
        endcase
    end
    always_ff @(posedge S_AXI_ACLK)
        if (!S_AXI_ARESETN) axi_rdata<=0;
        else if (slv_reg_rden) axi_rdata<=reg_data_out;

endmodule


