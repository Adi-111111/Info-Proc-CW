// rdp_v1_0_S00_AXI.sv
// --------------------
// AXI4-Lite slave for RDP simplification IP.
//
// Register map:
//   slv_reg0  0x00  CTRL:      [0]=start (auto-clears), [1]=done (read)
//   slv_reg1  0x04  IN_COUNT:  number of input points
//   slv_reg2  0x08  EPSILON:   RDP threshold Q10.6 (12.0 = 768)
//   slv_reg3  0x0C  OUT_COUNT: output point count (read only)
//   slv_reg4  0x10  BRAM_ADDR: BRAM address
//   slv_reg5  0x14  BRAM_DIN:  {x[15:0],y[15:0]} Q10.6 write to input BRAM
//   slv_reg6  0x18  BRAM_DOUT: read from output BRAM at BRAM_ADDR (read only)
//   slv_reg7  0x1C  BRAM_WE:   pulse 1 to write

`timescale 1ns/1ps

module rdp_v1_0_S00_AXI #(
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 5
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
    localparam integer ADDR_LSB          = 2;
    localparam integer OPT_MEM_ADDR_BITS = 2;

    logic [C_S_AXI_ADDR_WIDTH-1:0] axi_awaddr, axi_araddr;
    logic axi_awready, axi_wready, axi_bvalid, axi_arready, axi_rvalid, aw_en;
    logic [1:0] axi_bresp, axi_rresp;
    logic [C_S_AXI_DATA_WIDTH-1:0] axi_rdata, reg_data_out;
    logic [C_S_AXI_DATA_WIDTH-1:0] slv_reg0,slv_reg1,slv_reg2,slv_reg3;
    logic [C_S_AXI_DATA_WIDTH-1:0] slv_reg4,slv_reg5,slv_reg6,slv_reg7;
    logic slv_reg_rden, slv_reg_wren;

    assign S_AXI_AWREADY = axi_awready; assign S_AXI_WREADY  = axi_wready;
    assign S_AXI_BRESP   = axi_bresp;   assign S_AXI_BVALID  = axi_bvalid;
    assign S_AXI_ARREADY = axi_arready; assign S_AXI_RDATA   = axi_rdata;
    assign S_AXI_RRESP   = axi_rresp;   assign S_AXI_RVALID  = axi_rvalid;

    // ── User signals ────────────────────────────────────────────────────────
    logic        core_done;
    logic        done_latch;
    logic [7:0]  core_out_count;
    logic [31:0] in_bram[0:255], out_bram[0:255];
    logic [7:0]  core_in_addr, core_out_addr;
    logic [31:0] core_in_dout, core_out_din;
    logic        core_out_we;

    always_ff @(posedge S_AXI_ACLK)
        if (slv_reg_wren && axi_awaddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] == 3'h7)
            in_bram[slv_reg4[7:0]] <= slv_reg5;
    always_ff @(posedge S_AXI_ACLK)
        core_in_dout <= in_bram[core_in_addr];
    always_ff @(posedge S_AXI_ACLK)
        if (core_out_we) out_bram[core_out_addr] <= core_out_din;
    assign slv_reg6 = out_bram[slv_reg4[7:0]];

    (* DONT_TOUCH = "yes" *) rdp_core u_rdp (
        .clk          (S_AXI_ACLK),
        .rst_n        (S_AXI_ARESETN),
        .start        (slv_reg0[0]),
        .in_count     (slv_reg1[7:0]),
        .epsilon      (slv_reg2[15:0]),
        .in_bram_addr (core_in_addr),
        .in_bram_dout (core_in_dout),
        .out_bram_addr(core_out_addr),
        .out_bram_din (core_out_din),
        .out_bram_we  (core_out_we),
        .out_count    (core_out_count),
        .done         (core_done)
    );
    assign slv_reg3 = {24'b0, core_out_count};

    (* DONT_TOUCH = "yes" *) wire keep_core;
    assign keep_core = core_done ^ (|core_out_count) ^ (|core_in_addr)
                     ^ (|core_out_din) ^ core_out_we;

    // ── AXI write address ────────────────────────────────────────────────────
    always_ff @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin axi_awready<=0; aw_en<=1; end
        else begin
            if (!axi_awready&&S_AXI_AWVALID&&S_AXI_WVALID&&aw_en) begin
                axi_awready<=1; axi_awaddr<=S_AXI_AWADDR; aw_en<=0;
            end else if (S_AXI_BREADY&&axi_bvalid) begin aw_en<=1; axi_awready<=0; end
            else axi_awready<=0;
        end
    end

    always_ff @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) axi_wready<=0;
        else axi_wready <= (!axi_wready&&S_AXI_WVALID&&S_AXI_AWVALID&&aw_en);
    end

    assign slv_reg_wren = axi_wready&&S_AXI_WVALID&&axi_awready&&S_AXI_AWVALID;

    always_ff @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin slv_reg0<=0; done_latch<=0; end
        else begin
            slv_reg0 <= 0; // auto-clear start every cycle — single-cycle pulse
            if (core_done) done_latch <= 1'b1;
            if (slv_reg_wren&&axi_awaddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB]==3'h0)
                begin slv_reg0<=S_AXI_WDATA; done_latch<=1'b0; end
        end
    end

    always_ff @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            slv_reg1<=0; slv_reg2<=32'd768; // default epsilon=12.0 (12<<6)
            slv_reg4<=0; slv_reg5<=0; slv_reg7<=0;
        end else if (slv_reg_wren) begin
            case (axi_awaddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB])
                3'h1: slv_reg1<=S_AXI_WDATA;
                3'h2: slv_reg2<=S_AXI_WDATA;
                3'h4: slv_reg4<=S_AXI_WDATA;
                3'h5: slv_reg5<=S_AXI_WDATA;
                3'h7: slv_reg7<=S_AXI_WDATA;
                default:;
            endcase
        end else slv_reg7<=0;
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

    assign slv_reg_rden = axi_arready&S_AXI_ARVALID&~axi_rvalid;

    always_comb begin
        case (axi_araddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB])
            3'h0: reg_data_out = {30'b0, done_latch, slv_reg0[0]};
            3'h1: reg_data_out = slv_reg1;
            3'h2: reg_data_out = slv_reg2;
            3'h3: reg_data_out = slv_reg3;
            3'h4: reg_data_out = slv_reg4;
            3'h5: reg_data_out = slv_reg5;
            3'h6: reg_data_out = slv_reg6;
            3'h7: reg_data_out = slv_reg7;
            default: reg_data_out = 0;
        endcase
    end

    always_ff @(posedge S_AXI_ACLK)
        if (!S_AXI_ARESETN) axi_rdata<=0;
        else if (slv_reg_rden) axi_rdata<=reg_data_out;

endmodule


// ── Top-level wrapper ────────────────────────────────────────────────────────
