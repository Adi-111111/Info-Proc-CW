#include "kasa_driver.h"
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

void kasa_write(unsigned int BaseAddr, unsigned int offset, uint32_t data) {
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) return;
    void* map = mmap(NULL, MAP_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd, BaseAddr & ~MAP_MASK);
    close(fd);
    if (map == MAP_FAILED) return;
    *((volatile uint32_t*)((char*)map + (BaseAddr & MAP_MASK) + offset)) = data;
    munmap(map, MAP_SIZE);
}

uint32_t kasa_read(unsigned int BaseAddr, unsigned int offset) {
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) return 0;
    void* map = mmap(NULL, MAP_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd, BaseAddr & ~MAP_MASK);
    close(fd);
    if (map == MAP_FAILED) return 0;
    uint32_t val = *((volatile uint32_t*)((char*)map + (BaseAddr & MAP_MASK) + offset));
    munmap(map, MAP_SIZE);
    return val;
}

int kasa_circle_fit(unsigned int BaseAddr,
                    float *in_x, float *in_y, int in_n,
                    float *cx, float *cy, float *r) {

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) return 0;
    void* map = mmap(NULL, MAP_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd, BaseAddr & ~MAP_MASK);
    close(fd);
    if (map == MAP_FAILED) return 0;

    volatile uint32_t* reg = (volatile uint32_t*)((char*)map + (BaseAddr & MAP_MASK));
    int n = in_n > 256 ? 256 : in_n;

    for (int i = 0; i < n; i++) {
        reg[KASA_BRAM_ADDR/4] = (uint32_t)i;
        reg[KASA_BRAM_DIN/4]  = pack_xy(in_x[i], in_y[i]);
        reg[KASA_BRAM_WE/4]   = 1;
    }
    reg[KASA_IN_COUNT/4] = (uint32_t)n;
    reg[KASA_CTRL/4]     = 1;

    int timeout = POLL_TIMEOUT;
    while (!(reg[KASA_CTRL/4] & 0x2) && --timeout > 0);

    int valid = 0;
    if (timeout > 0) {
        valid = (int)(reg[KASA_VALID/4] & 0x1);
        if (valid) {
            *cx = fp_to_float((uint16_t)(reg[KASA_CX_OUT/4] & 0xFFFF));
            *cy = fp_to_float((uint16_t)(reg[KASA_CY_OUT/4] & 0xFFFF));
            *r  = fp_to_float((uint16_t)(reg[KASA_R_OUT/4]  & 0xFFFF));
        }
    }

    munmap(map, MAP_SIZE);
    return valid;
}
