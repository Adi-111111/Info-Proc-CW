module kasa_v1_0 #(
    parameter integer C_S00_AXI_DATA_WIDTH = 32,
    parameter integer C_S00_AXI_ADDR_WIDTH = 5
)(
    input  logic                                 s00_axi_aclk,
    input  logic                                 s00_axi_aresetn,
    input  logic [C_S00_AXI_ADDR_WIDTH-1:0]     s00_axi_awaddr,
    input  logic [2:0]                           s00_axi_awprot,
    input  logic                                 s00_axi_awvalid,
    output logic                                 s00_axi_awready,
    input  logic [C_S00_AXI_DATA_WIDTH-1:0]     s00_axi_wdata,
    input  logic [(C_S00_AXI_DATA_WIDTH/8)-1:0] s00_axi_wstrb,
    input  logic                                 s00_axi_wvalid,
    output logic                                 s00_axi_wready,
    output logic [1:0]                           s00_axi_bresp,
    output logic                                 s00_axi_bvalid,
    input  logic                                 s00_axi_bready,
    input  logic [C_S00_AXI_ADDR_WIDTH-1:0]     s00_axi_araddr,
    input  logic [2:0]                           s00_axi_arprot,
    input  logic                                 s00_axi_arvalid,
    output logic                                 s00_axi_arready,
    output logic [C_S00_AXI_DATA_WIDTH-1:0]     s00_axi_rdata,
    output logic [1:0]                           s00_axi_rresp,
    output logic                                 s00_axi_rvalid,
    input  logic                                 s00_axi_rready
);
    localparam integer FP_WIDTH = 16;
    localparam integer FP_FRAC  = 6;
    kasa_v1_0_S00_AXI #(
        .C_S_AXI_DATA_WIDTH(C_S00_AXI_DATA_WIDTH),
        .C_S_AXI_ADDR_WIDTH(C_S00_AXI_ADDR_WIDTH)
    ) inst (
        .S_AXI_ACLK(s00_axi_aclk), .S_AXI_ARESETN(s00_axi_aresetn),
        .S_AXI_AWADDR(s00_axi_awaddr), .S_AXI_AWPROT(s00_axi_awprot),
        .S_AXI_AWVALID(s00_axi_awvalid), .S_AXI_AWREADY(s00_axi_awready),
        .S_AXI_WDATA(s00_axi_wdata), .S_AXI_WSTRB(s00_axi_wstrb),
        .S_AXI_WVALID(s00_axi_wvalid), .S_AXI_WREADY(s00_axi_wready),
        .S_AXI_BRESP(s00_axi_bresp), .S_AXI_BVALID(s00_axi_bvalid),
        .S_AXI_BREADY(s00_axi_bready), .S_AXI_ARADDR(s00_axi_araddr),
        .S_AXI_ARPROT(s00_axi_arprot), .S_AXI_ARVALID(s00_axi_arvalid),
        .S_AXI_ARREADY(s00_axi_arready), .S_AXI_RDATA(s00_axi_rdata),
        .S_AXI_RRESP(s00_axi_rresp), .S_AXI_RVALID(s00_axi_rvalid),
        .S_AXI_RREADY(s00_axi_rready)
    );
endmodule
