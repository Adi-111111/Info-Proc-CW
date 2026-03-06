#ifndef RESAMPLE_DRIVER_H
#define RESAMPLE_DRIVER_H
#include <stdint.h>

#define RESAMPLE_CTRL       0x00
#define RESAMPLE_IN_COUNT   0x04
#define RESAMPLE_STEP       0x08
#define RESAMPLE_OUT_COUNT  0x0C
#define RESAMPLE_BRAM_ADDR  0x10
#define RESAMPLE_BRAM_DIN   0x14
#define RESAMPLE_BRAM_DOUT  0x18
#define RESAMPLE_BRAM_WE    0x1C
#define FP_SCALE 64

#ifdef __cplusplus
extern "C" {
#endif

void     resample_write(unsigned int BaseAddr, unsigned int offset, uint32_t data);
uint32_t resample_read (unsigned int BaseAddr, unsigned int offset);
int      resample(unsigned int BaseAddr,
                  float *in_x, float *in_y, int in_n,
                  float step,
                  float *out_x, float *out_y);

#ifdef __cplusplus
}
#endif
#endif
