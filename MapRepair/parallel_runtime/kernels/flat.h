#pragma once
#include <cmath>
#include <cstdint>
#ifdef __HIPCC__
#define ADM_FLAT_HD __host__ __device__
#else
#define ADM_FLAT_HD
#endif
// A plane is affine in inverse camera depth. Reject every uncertain sample.
// This classifier is a heuristic, NOT a proof that skipping refinement is exact.
ADM_FLAT_HD inline bool adm_flat_sample(const float* d, const float* c, int w,
    int x0, int y0, int x, int y) {
    float a=d[y0*w+x0], b=d[y0*w+x0+31], e=d[(y0+31)*w+x0];
    float z=d[y*w+x], score=c[y*w+x];
    if (!(a>0 && b>0 && e>0 && z>0 && a<1e7f && b<1e7f && e<1e7f && z<1e7f
        && score>=0 && score<=0.03f)) return false;
    float p=1.f/a;
    float dx=(1.f/b-p)*(float(x-x0)/31.f);
    float dy=(1.f/e-p)*(float(y-y0)/31.f);
    float predicted=(p+dx)+dy;
    float actual=1.f/z;
    return fabsf(actual-predicted)<=actual*0.00005f;
}
inline void adm_flat_cpu(const float* d,const float* c,int w,int h,uint8_t* out) {
    int tw=(w+31)/32, th=(h+31)/32;
    for(int ty=0;ty<th;++ty) for(int tx=0;tx<tw;++tx) {
        bool ok=(tx*32+32<=w && ty*32+32<=h);
        for(int y=ty*32;ok && y<ty*32+32;++y)
            for(int x=tx*32;ok && x<tx*32+32;++x) ok=adm_flat_sample(d,c,w,tx*32,ty*32,x,y);
        out[ty*tw+tx]=ok;
    }
}
inline bool adm_flat_interior(const uint8_t* m,int tw,int th,int x,int y) {
    if(x<1 || y<1 || x>=tw-1 || y>=th-1) return false;
    for(int j=-1;j<=1;++j) for(int i=-1;i<=1;++i) if(!m[(y+j)*tw+x+i]) return false;
    return true;
}
