#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include "composite.h"
#include "flat.h"
namespace {
thread_local std::string error;
void check(hipError_t e) { if(e!=hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
constexpr size_t capacity=1024*1024;
__global__ void composite(const uint32_t* src,uint32_t* dst,size_t n,uint32_t bg) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n) dst[i]=adm_pixel(src[i],bg);
}
__global__ void flat(const float* d,const float* c,int w,int h,uint8_t* out) {
    __shared__ unsigned invalid;
    if(threadIdx.x==0) invalid=0;
    __syncthreads();
    int x0=blockIdx.x*32,y0=blockIdx.y*32;
    bool ok=(x0+32<=w && y0+32<=h);
    if(ok) for(int k=threadIdx.x;k<1024;k+=blockDim.x)
        if(!adm_flat_sample(d,c,w,x0,y0,x0+k%32,y0+k/32)) ok=false;
    if(!ok) atomicOr(&invalid,1u);
    __syncthreads();
    if(threadIdx.x==0) out[blockIdx.y*gridDim.x+blockIdx.x]=invalid?0:1;
}
struct Slot {
    hipStream_t stream=nullptr;
    uint32_t *in=nullptr,*out=nullptr,*host_in=nullptr,*host_out=nullptr;
    void clear() {
        if(stream) hipStreamSynchronize(stream);
        if(in) hipFree(in); if(out) hipFree(out);
        if(host_in) hipHostFree(host_in); if(host_out) hipHostFree(host_out);
        if(stream) hipStreamDestroy(stream);
        stream=nullptr; in=out=host_in=host_out=nullptr;
    }
    void init() {
        check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
        check(hipMalloc(&in,capacity*4)); check(hipMalloc(&out,capacity*4));
        check(hipHostMalloc(&host_in,capacity*4)); check(hipHostMalloc(&host_out,capacity*4));
    }
    ~Slot(){clear();}
};
struct Context {
    std::mutex mutex; Slot slots[2]; bool ready=false;
    void init() {
        if(ready)return;
        int count=0;check(hipGetDeviceCount(&count));
        if(!count)throw std::runtime_error("No HIP device");
        check(hipSetDevice(0));
        try {for(auto& s:slots)s.init();ready=true;}
        catch(...) {for(auto& s:slots)s.clear();throw;}
    }
};
Context ctx;
}
extern "C" const char* adm_hip_error(){return error.c_str();}
extern "C" int adm_hip_composite(const uint32_t* src,uint32_t* dst,size_t n,uint32_t bg) {
    std::lock_guard<std::mutex> lock(ctx.mutex);
    try {
        if(!src||!dst||n>16777216)throw std::runtime_error("Invalid or oversized composite");
        ctx.init(); check(hipSetDevice(0));
        size_t offsets[2]={0,0}, counts[2]={0,0};
        for(size_t offset=0,step=0;offset<n;offset+=capacity,++step) {
            int j=step%2;auto& s=ctx.slots[j];
            check(hipStreamSynchronize(s.stream));
            if(counts[j])std::memcpy(dst+offsets[j],s.host_out,counts[j]*4);
            size_t size=std::min(capacity,n-offset); offsets[j]=offset;counts[j]=size;
            std::memcpy(s.host_in,src+offset,size*4);
            check(hipMemcpyAsync(s.in,s.host_in,size*4,hipMemcpyHostToDevice,s.stream));
            hipLaunchKernelGGL(composite,dim3((size+255)/256),dim3(256),0,s.stream,s.in,s.out,size,bg);
            check(hipGetLastError());
            check(hipMemcpyAsync(s.host_out,s.out,size*4,hipMemcpyDeviceToHost,s.stream));
        }
        for(int j=0;j<2;++j) {auto& s=ctx.slots[j];check(hipStreamSynchronize(s.stream));
            if(counts[j])std::memcpy(dst+offsets[j],s.host_out,counts[j]*4);}
        return 0;
    } catch(const std::exception& e) {
        error=e.what();for(auto& s:ctx.slots)s.clear();ctx.ready=false;return -1;
    }
}
extern "C" int adm_flat_tiles(const float* d,const float* c,int w,int h,uint8_t* out) {
    std::lock_guard<std::mutex> lock(ctx.mutex);
    float *dd=nullptr,*dc=nullptr;uint8_t* mask=nullptr;hipStream_t stream=nullptr;
    int status=0;
    try {
        if(!d||!c||!out||w<1||h<1||w>4096||h>4096)throw std::runtime_error("Invalid flat-map size");
        check(hipSetDevice(0));
        size_t bytes=size_t(w)*h*4, tiles=size_t((w+31)/32)*((h+31)/32);
        check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
        check(hipMalloc(&dd,bytes));check(hipMalloc(&dc,bytes));check(hipMalloc(&mask,tiles));
        check(hipMemcpyAsync(dd,d,bytes,hipMemcpyHostToDevice,stream));
        check(hipMemcpyAsync(dc,c,bytes,hipMemcpyHostToDevice,stream));
        hipLaunchKernelGGL(flat,dim3((w+31)/32,(h+31)/32),dim3(256),0,stream,dd,dc,w,h,mask);
        check(hipGetLastError());check(hipMemcpyAsync(out,mask,tiles,hipMemcpyDeviceToHost,stream));
        check(hipStreamSynchronize(stream));
    } catch(const std::exception& e) {error=e.what();status=-1;}
    if(stream)hipStreamSynchronize(stream);
    if(dd)hipFree(dd);if(dc)hipFree(dc);if(mask)hipFree(mask);if(stream)hipStreamDestroy(stream);
    return status;
}
