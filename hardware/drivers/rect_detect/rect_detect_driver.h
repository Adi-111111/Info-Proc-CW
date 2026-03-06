#ifndef RECT_DETECT_DRIVER_H
#define RECT_DETECT_DRIVER_H
#include <stdint.h>

#define RECT_CTRL        0x00
#define RECT_CORNER0_X   0x04
#define RECT_CORNER0_Y   0x08
#define RECT_CORNER1_X   0x0C
#define RECT_CORNER1_Y   0x10
#define RECT_CORNER2_X   0x14
#define RECT_CORNER2_Y   0x18
#define RECT_CORNER3_X   0x1C
#define RECT_CORNER3_Y   0x20
#define RECT_ANGLE_TOL   0x24
#define RECT_RESULT      0x28
#define FP_SCALE 64

#ifdef __cplusplus
extern "C" {
#endif

void     rect_write(unsigned int BaseAddr, unsigned int offset, uint32_t data);
uint32_t rect_read (unsigned int BaseAddr, unsigned int offset);
int      rect_detect(unsigned int BaseAddr,
                     float *corners_x, float *corners_y,
                     float angle_tol);

#ifdef __cplusplus
}
#endif
#endif
