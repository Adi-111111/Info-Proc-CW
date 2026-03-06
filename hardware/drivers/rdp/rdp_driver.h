#ifndef RDP_DRIVER_H
#define RDP_DRIVER_H
#include <stdint.h>

#define RDP_CTRL       0x00
#define RDP_IN_COUNT   0x04
#define RDP_EPSILON    0x08
#define RDP_OUT_COUNT  0x0C
#define RDP_BRAM_ADDR  0x10
#define RDP_BRAM_DIN   0x14
#define RDP_BRAM_DOUT  0x18
#define RDP_BRAM_WE    0x1C
#define FP_SCALE 64

#ifdef __cplusplus
extern "C" {
#endif

void     rdp_write(unsigned int BaseAddr, unsigned int offset, uint32_t data);
uint32_t rdp_read (unsigned int BaseAddr, unsigned int offset);
int      rdp_simplify(unsigned int BaseAddr,
                      float *in_x, float *in_y, int in_n,
                      float epsilon,
                      float *out_x, float *out_y);

#ifdef __cplusplus
}
#endif
#endif
