#include "rdp_driver.h"
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
static inline float fp_to_float(uint16_t v) {
    return (float)(int16_t)v / 64.0f;
}
static inline uint32_t pack_xy(float x, float y) {
    return ((uint32_t)float_to_fp(x) << 16) | (uint32_t)float_to_fp(y);
}

void rdp_write(unsigned int BaseAddr, unsigned int offset, uint32_t data) {
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) return;
    void* map = mmap(NULL, MAP_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd, BaseAddr & ~MAP_MASK);
    close(fd);
    if (map == MAP_FAILED) return;
    *((volatile uint32_t*)((char*)map + (BaseAddr & MAP_MASK) + offset)) = data;
    munmap(map, MAP_SIZE);
}

uint32_t rdp_read(unsigned int BaseAddr, unsigned int offset) {
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) return 0;
    void* map = mmap(NULL, MAP_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd, BaseAddr & ~MAP_MASK);
    close(fd);
    if (map == MAP_FAILED) return 0;
    uint32_t val = *((volatile uint32_t*)((char*)map + (BaseAddr & MAP_MASK) + offset));
    munmap(map, MAP_SIZE);
    return val;
}

int rdp_simplify(unsigned int BaseAddr,
                 float *in_x, float *in_y, int in_n,
                 float epsilon,
                 float *out_x, float *out_y) {

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) return 0;
    void* map = mmap(NULL, MAP_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd, BaseAddr & ~MAP_MASK);
    close(fd);
    if (map == MAP_FAILED) return 0;

    volatile uint32_t* reg = (volatile uint32_t*)((char*)map + (BaseAddr & MAP_MASK));
    int n = in_n > 256 ? 256 : in_n;

    for (int i = 0; i < n; i++) {
        reg[RDP_BRAM_ADDR/4] = (uint32_t)i;
        reg[RDP_BRAM_DIN/4]  = pack_xy(in_x[i], in_y[i]);
        reg[RDP_BRAM_WE/4]   = 1;
    }
    reg[RDP_IN_COUNT/4] = (uint32_t)n;
    reg[RDP_EPSILON/4]  = (uint32_t)float_to_fp(epsilon);
    reg[RDP_CTRL/4]     = 1;

    int timeout = POLL_TIMEOUT;
    while (!(reg[RDP_CTRL/4] & 0x2) && --timeout > 0);

    int out_n = 0;
    if (timeout > 0) {
        out_n = (int)(reg[RDP_OUT_COUNT/4] & 0xFF);
        for (int i = 0; i < out_n; i++) {
            reg[RDP_BRAM_ADDR/4] = (uint32_t)i;
            uint32_t word = reg[RDP_BRAM_DOUT/4];
            out_x[i] = fp_to_float((uint16_t)(word >> 16));
            out_y[i] = fp_to_float((uint16_t)(word & 0xFFFF));
        }
    }

    munmap(map, MAP_SIZE);
    return out_n;
}
