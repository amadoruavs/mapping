#include <stdint.h>
// Peano/LLVM vector operations: 16 packed BGRA pixels (512 bits) per iteration.
// ObjectFifo tiles have a multiple of 16 elements. The host pads only the tail.
#ifndef ADM_BACKGROUND
#define ADM_BACKGROUND 0x00ffffffu
#endif
typedef uint32_t adm_v16 __attribute__((vector_size(64)));
extern "C" void adm_xdna_composite(const uint32_t* __restrict src,
                                  uint32_t* __restrict dst, int32_t n) {
    for(int32_t i=0;i<n;i+=16) {
        adm_v16 p; __builtin_memcpy(&p,src+i,sizeof(p));
        adm_v16 a=p>>24;
        adm_v16 t0=(p&255)*a+(ADM_BACKGROUND&255)*(255-a)+127;
        adm_v16 t1=((p>>8)&255)*a+((ADM_BACKGROUND>>8)&255)*(255-a)+127;
        adm_v16 t2=((p>>16)&255)*a+((ADM_BACKGROUND>>16)&255)*(255-a)+127;
        adm_v16 q0=(t0+1+(t0>>8))>>8, q1=(t1+1+(t1>>8))>>8, q2=(t2+1+(t2>>8))>>8;
        adm_v16 out=q0|(q1<<8)|(q2<<16)|0xff000000u;
        __builtin_memcpy(dst+i,&out,sizeof(out));
    }
}
