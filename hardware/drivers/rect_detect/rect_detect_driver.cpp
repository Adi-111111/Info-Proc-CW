#include "rect_detect_driver.h"
#include <stdint.h>
#include <math.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#define MAP_SIZE 0x10000UL
#define MAP_MASK (MAP_SIZE - 1)
#define POLL_TIMEOUT 1000000

static inline uint16_t float_to_fp(float x) {
    int v = (int)roundf(x * 64.0f);
    return (uint16_t)(v & 0xFFFF);
}

void rect_write(unsigned int BaseAddr, unsigned int offset, uint32_t data) {
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) return;
    void* map = mmap(NULL, MAP_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd, BaseAddr & ~MAP_MASK);
    close(fd);
    if (map == MAP_FAILED) return;
    *((volatile uint32_t*)((char*)map + (BaseAddr & MAP_MASK) + offset)) = data;
    munmap(map, MAP_SIZE);
}

uint32_t rect_read(unsigned int BaseAddr, unsigned int offset) {
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) return 0;
    void* map = mmap(NULL, MAP_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd, BaseAddr & ~MAP_MASK);
    close(fd);
    if (map == MAP_FAILED) return 0;
    uint32_t val = *((volatile uint32_t*)((char*)map + (BaseAddr & MAP_MASK) + offset));
    munmap(map, MAP_SIZE);
    return val;
}

int rect_detect(unsigned int BaseAddr,
                float *corners_x, float *corners_y,
                float angle_tol) {

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) return 0;
    void* map = mmap(NULL, MAP_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd, BaseAddr & ~MAP_MASK);
    close(fd);
    if (map == MAP_FAILED) return 0;

    volatile uint32_t* reg = (volatile uint32_t*)((char*)map + (BaseAddr & MAP_MASK));

    reg[RECT_CORNER0_X/4] = (uint32_t)float_to_fp(corners_x[0]);
    reg[RECT_CORNER0_Y/4] = (uint32_t)float_to_fp(corners_y[0]);
    reg[RECT_CORNER1_X/4] = (uint32_t)float_to_fp(corners_x[1]);
    reg[RECT_CORNER1_Y/4] = (uint32_t)float_to_fp(corners_y[1]);
    reg[RECT_CORNER2_X/4] = (uint32_t)float_to_fp(corners_x[2]);
    reg[RECT_CORNER2_Y/4] = (uint32_t)float_to_fp(corners_y[2]);
    reg[RECT_CORNER3_X/4] = (uint32_t)float_to_fp(corners_x[3]);
    reg[RECT_CORNER3_Y/4] = (uint32_t)float_to_fp(corners_y[3]);
    reg[RECT_ANGLE_TOL/4] = (uint32_t)float_to_fp(angle_tol);
    reg[RECT_CTRL/4]      = 1;

    int timeout = POLL_TIMEOUT;
    while (!(reg[RECT_CTRL/4] & 0x2) && --timeout > 0);

    int result = 0;
    if (timeout > 0)
        result = (int)(reg[RECT_RESULT/4] & 0x1);

    munmap(map, MAP_SIZE);
    return result;
}
