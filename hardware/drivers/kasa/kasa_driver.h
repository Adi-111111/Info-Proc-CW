#ifndef KASA_DRIVER_H
#define KASA_DRIVER_H
#include <stdint.h>

#define KASA_CTRL      0x00
#define KASA_IN_COUNT  0x04
#define KASA_CX_OUT    0x08
#define KASA_CY_OUT    0x0C
#define KASA_R_OUT     0x10
#define KASA_VALID     0x14
#define KASA_BRAM_ADDR 0x18
#define KASA_BRAM_DIN  0x1C
#define KASA_BRAM_WE   0x20
#define FP_SCALE 64

#ifdef __cplusplus
extern "C" {
#endif

void     kasa_write(unsigned int BaseAddr, unsigned int offset, uint32_t data);
uint32_t kasa_read (unsigned int BaseAddr, unsigned int offset);
int      kasa_circle_fit(unsigned int BaseAddr,
                         float *in_x, float *in_y, int in_n,
                         float *cx, float *cy, float *r);

#ifdef __cplusplus
}
#endif
#endif
