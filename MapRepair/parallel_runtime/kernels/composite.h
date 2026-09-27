#pragma once
#include <stdint.h>
#ifdef __HIPCC__
#define ADM_HD __host__ __device__
#else
#define ADM_HD
#endif
// Exact division by 255 for 0 <= t <= 65152. No floating point or quantization.
ADM_HD inline uint32_t adm_channel(uint32_t c, uint32_t a, uint32_t b) {
    uint32_t t = c*a + b*(255-a) + 127;
    return (t + 1 + (t >> 8)) >> 8;
}
ADM_HD inline uint32_t adm_pixel(uint32_t p, uint32_t bg) {
    const uint32_t a=p>>24;
    return adm_channel(p&255,a,bg&255) |
        (adm_channel((p>>8)&255,a,(bg>>8)&255)<<8) |
        (adm_channel((p>>16)&255,a,(bg>>16)&255)<<16) | 0xff000000u;
}
