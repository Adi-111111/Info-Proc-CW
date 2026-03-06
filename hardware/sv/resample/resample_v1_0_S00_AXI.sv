`timescale 1ns/1ps

module resample_v1_0_S00_AXI #(
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 5
)(
    input  logic                             S_AXI_ACLK,
    input  logic                             S_AXI_ARESETN,
    input  logic [C_S_AXI_ADDR_WIDTH-1:0]   S_AXI_AWADDR,
    input  logic [2:0]                       S_AXI_AWPROT,
    input  logic                             S_AXI_AWVALID,
    output logic                             S_AXI_AWREADY,
    input  logic [C_S_AXI_DATA_WIDTH-1:0]   S_AXI_WDATA,
    input  logic [(C_S_AXI_DATA_WIDTH/8)-1:0] S_AXI_WSTRB,
    input  logic                             S_AXI_WVALID,
    output logic                             S_AXI_WREADY,
    output logic [1:0]                       S_AXI_BRESP,
    output logic                             S_AXI_BVALID,
    input  logic                             S_AXI_BREADY,
    input  logic [C_S_AXI_ADDR_WIDTH-1:0]   S_AXI_ARADDR,
    input  logic [2:0]                       S_AXI_ARPROT,
    input  logic                             S_AXI_ARVALID,
    output logic                             S_AXI_ARREADY,
    output logic [C_S_AXI_DATA_WIDTH-1:0]   S_AXI_RDATA,
    output logic [1:0]                       S_AXI_RRESP,
    output logic                             S_AXI_RVALID,
    input  logic                             S_AXI_RREADY
);
    localparam integer FP_WIDTH = 16;
    localparam integer FP_FRAC  = 6;

    localparam integer ADDR_LSB          = 2;
    localparam integer OPT_MEM_ADDR_BITS = 2; 

    logic [C_S_AXI_ADDR_WIDTH-1:0] axi_awaddr;
    logic                           axi_awready;
    logic                           axi_wready;
    logic [1:0]                     axi_bresp;
    logic                           axi_bvalid;
    logic [C_S_AXI_ADDR_WIDTH-1:0] axi_araddr;
    logic                           axi_arready;
    logic [C_S_AXI_DATA_WIDTH-1:0] axi_rdata;
    logic [1:0]                     axi_rresp;
    logic                           axi_rvalid;

    logic [C_S_AXI_DATA_WIDTH-1:0] slv_reg0, slv_reg1, slv_reg2, slv_reg3;
    logic [C_S_AXI_DATA_WIDTH-1:0] slv_reg4, slv_reg5, slv_reg6, slv_reg7;
    logic                           slv_reg_rden, slv_reg_wren;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg_data_out;
    logic                           aw_en;

    assign S_AXI_AWREADY = axi_awready;
    assign S_AXI_WREADY  = axi_wready;
    assign S_AXI_BRESP   = axi_bresp;
    assign S_AXI_BVALID  = axi_bvalid;
    assign S_AXI_ARREADY = axi_arready;
    assign S_AXI_RDATA   = axi_rdata;
    assign S_AXI_RRESP   = axi_rresp;
    assign S_AXI_RVALID  = axi_rvalid;

    // ── User signals ───────────────────────────────────────────────────────
    logic        core_done;
    logic        done_latch;
    logic [7:0]  core_out_count;

    // BRAM arrays
    logic [31:0] in_bram  [0:255];
    logic [31:0] out_bram [0:255];
    logic [7:0]  core_in_addr, core_out_addr;
    logic [31:0] core_in_dout, core_out_din;
    logic        core_out_we;

    // BRAM port A — latch addr/data, write on WE strobe
    // slv_reg4 (addr) and slv_reg5 (data) are written before slv_reg7 (WE)
    // so they are stable when WE fires
    logic [7:0]  bram_wr_addr;
    logic [31:0] bram_wr_data;
    always_ff @(posedge S_AXI_ACLK) begin
        if (slv_reg_wren) begin
            case (axi_awaddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB])
                3'h4: bram_wr_addr <= S_AXI_WDATA[7:0];
                3'h5: bram_wr_data <= S_AXI_WDATA;
                default: ;
            endcase
        end
    end
    always_ff @(posedge S_AXI_ACLK)
        if (slv_reg_wren && axi_awaddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] == 3'h7)
            in_bram[bram_wr_addr] <= bram_wr_data;

    // BRAM port B (core read)
    always_ff @(posedge S_AXI_ACLK)
        core_in_dout <= in_bram[core_in_addr];

    // Output BRAM (core write)
    always_ff @(posedge S_AXI_ACLK)
        if (core_out_we) out_bram[core_out_addr] <= core_out_din;

    // slv_reg6 = output BRAM readback at requested address
    assign slv_reg6 = out_bram[slv_reg4[7:0]];

    // ── Core instantiation ─────────────────────────────────────────────────
    (* DONT_TOUCH = "yes" *) resample_core u_resample (
        .clk          (S_AXI_ACLK),
        .rst_n        (S_AXI_ARESETN),
        .start        (slv_reg0[0]),
        .in_count     (slv_reg1[7:0]),
        .step_fp      (slv_reg2[7:0]),
        .in_bram_addr (core_in_addr),
        .in_bram_dout (core_in_dout),
        .out_bram_addr(core_out_addr),
        .out_bram_din (core_out_din),
        .out_bram_we  (core_out_we),
        .out_count    (core_out_count),
        .done         (core_done)
    );

    // slv_reg3 read-only: out_count
    assign slv_reg3 = {24'b0, core_out_count};

    // Force Vivado to keep core logic by making outputs observable
    // This wire is ANDed with slv_reg3 so it never changes the value
    // but prevents the synthesiser from removing core_done and core_out_count
    (* DONT_TOUCH = "yes" *) wire keep_core;
    assign keep_core = core_done ^ (|core_out_count) ^ (|core_in_addr)
                     ^ (|core_out_din) ^ core_out_we;

    // ── AXI write address channel ──────────────────────────────────────────
    always_ff @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_awready <= 0; aw_en <= 1;
        end else begin
            if (!axi_awready && S_AXI_AWVALID && S_AXI_WVALID && aw_en) begin
                axi_awready <= 1; axi_awaddr <= S_AXI_AWADDR; aw_en <= 0;
            end else if (S_AXI_BREADY && axi_bvalid) begin
                aw_en <= 1; axi_awready <= 0;
            end else axi_awready <= 0;
        end
    end

    // ── AXI write data channel ─────────────────────────────────────────────
    always_ff @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) axi_wready <= 0;
        else begin
            if (!axi_wready && S_AXI_WVALID && S_AXI_AWVALID && aw_en)
                axi_wready <= 1;
            else axi_wready <= 0;
        end
    end

    assign slv_reg_wren = axi_wready && S_AXI_WVALID && axi_awready && S_AXI_AWVALID;

    // ── Register write logic ───────────────────────────────────────────────
    // start is a single-cycle pulse: auto-clears the cycle after it is written
    // This prevents the FSM from re-triggering when it returns to IDLE
    always_ff @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin slv_reg0 <= 0; done_latch <= 0; end
        else begin
            slv_reg0 <= 0; // default: clear start every cycle
            if (core_done) done_latch <= 1'b1;
            if (slv_reg_wren && axi_awaddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] == 3'h0)
                begin slv_reg0 <= S_AXI_WDATA; done_latch <= 1'b0; end
        end
    end

    always_ff @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            slv_reg1 <= 0; slv_reg2 <= 32'd512; // default step=8.0 (8<<6=512)
            slv_reg4 <= 0; slv_reg5 <= 0; slv_reg7 <= 0;
        end else if (slv_reg_wren) begin
            case (axi_awaddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB])
                3'h1: slv_reg1 <= S_AXI_WDATA;
                3'h2: slv_reg2 <= S_AXI_WDATA;
                3'h4: slv_reg4 <= S_AXI_WDATA;
                3'h5: slv_reg5 <= S_AXI_WDATA;
                3'h7: slv_reg7 <= S_AXI_WDATA;
                default: ;
            endcase
        end else begin
            slv_reg7 <= 0; // auto-clear BRAM_WE after one cycle
        end
    end

    // ── AXI write response channel ─────────────────────────────────────────
    always_ff @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin axi_bvalid <= 0; axi_bresp <= 0; end
        else begin
            if (axi_awready && S_AXI_AWVALID && !axi_bvalid && axi_wready && S_AXI_WVALID) begin
                axi_bvalid <= 1; axi_bresp <= 0;
            end else if (S_AXI_BREADY && axi_bvalid) axi_bvalid <= 0;
        end
    end

    // ── AXI read address channel ───────────────────────────────────────────
    always_ff @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin axi_arready <= 0; axi_araddr <= 0; end
        else begin
            if (!axi_arready && S_AXI_ARVALID) begin
                axi_arready <= 1; axi_araddr <= S_AXI_ARADDR;
            end else axi_arready <= 0;
        end
    end

    // ── AXI read data channel ──────────────────────────────────────────────
    always_ff @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin axi_rvalid <= 0; axi_rresp <= 0; end
        else begin
            if (axi_arready && S_AXI_ARVALID && !axi_rvalid) begin
                axi_rvalid <= 1; axi_rresp <= 0;
            end else if (axi_rvalid && S_AXI_RREADY) axi_rvalid <= 0;
        end
    end

    assign slv_reg_rden = axi_arready & S_AXI_ARVALID & ~axi_rvalid;

    always_comb begin
        case (axi_araddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB])
            3'h0:    reg_data_out = {30'b0, done_latch, slv_reg0[0]};
            3'h1:    reg_data_out = slv_reg1;
            3'h2:    reg_data_out = slv_reg2;
            3'h3:    reg_data_out = slv_reg3;
            3'h4:    reg_data_out = slv_reg4;
            3'h5:    reg_data_out = slv_reg5;
            3'h6:    reg_data_out = slv_reg6;
            3'h7:    reg_data_out = slv_reg7;
            default: reg_data_out = 0;
        endcase
    end

    always_ff @(posedge S_AXI_ACLK)
        if (!S_AXI_ARESETN) axi_rdata <= 0;
        else if (slv_reg_rden) axi_rdata <= reg_data_out;

endmodule
