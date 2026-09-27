// AMD FidelityFX FSR1 EASU/RCAS FP32 equations, adapted for Metal buffers.
// Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
// MIT license: see ../third_party/fidelityfx_fsr1/LICENSE (included below).
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

inline float4 fsr_load(device const uchar *src, int2 p, uint w, uint h, uint channels) {
    p=clamp(p,int2(0),int2(w-1,h-1)); uint k=(p.y*w+p.x)*channels;
    return float4(src[k],src[k+1],src[k+2],channels==4?src[k+3]:255)/255.f;
}
inline void fsr_store(device uchar *dst,uint i,uint channels,float4 value) {
    uchar4 c=uchar4(rint(clamp(value,0.f,1.f)*255.f));uint k=i*channels;
    dst[k]=c.x;dst[k+1]=c.y;dst[k+2]=c.z;if(channels==4)dst[k+3]=c.w;
}
inline float fsr_luma(float3 c) { return .5f*c.x+c.y+.5f*c.z; }
inline float4 fsr_bilinear(device const uchar *src,float2 xy,uint w,uint h,uint channels) {
    int2 b=int2(floor(xy));float2 f=xy-float2(b);
    return mix(mix(fsr_load(src,b,w,h,channels),fsr_load(src,b+int2(1,0),w,h,channels),f.x),
        mix(fsr_load(src,b+int2(0,1),w,h,channels),fsr_load(src,b+int2(1,1),w,h,channels),f.x),f.y);
}
inline void fsr_set(thread float2 &dir,thread float &len,float weight,float a,float b,float c,float d,float e) {
    float2 gradient=float2(d-b,e-a);
    float2 magnitude=max(abs(float2(d-c,e-c)),abs(float2(c-b,c-a)));
    float2 edge=clamp(abs(gradient)/max(magnitude,float2(1e-8f)),0.f,1.f);
    dir+=gradient*weight;len+=dot(edge,edge)*weight;
}
inline float4 fsr_easu_sample(device const uchar *src,float2 xy,uint w,uint h,uint channels) {
    int2 base=int2(floor(xy));float2 pp=xy-float2(base);
    const int2 offsets[12]={int2(0,-1),int2(1,-1),int2(-1,1),int2(0,1),
        int2(0,0),int2(-1,0),int2(1,1),int2(2,1),int2(2,0),int2(1,0),int2(1,2),int2(0,2)};
    float4 taps[12];float l[12];float alpha=1.f;
    for(uint k=0;k<12;k++){taps[k]=fsr_load(src,base+offsets[k],w,h,channels);l[k]=fsr_luma(taps[k].xyz);alpha=min(alpha,taps[k].w);}
    // ADM: avoid filtering a transparent map boundary into visible colors.
    if(alpha<1.f){
        float weights[4]={(1-pp.x)*(1-pp.y),pp.x*(1-pp.y),(1-pp.x)*pp.y,pp.x*pp.y};
        uint corners[4]={4,9,3,6};float3 c=0;float sum=0;
        for(uint k=0;k<4;k++){float q=weights[k]*taps[corners[k]].w;c+=taps[corners[k]].xyz*q;sum+=q;}
        float a=fsr_load(src,int2(floor(xy+.5f)),w,h,channels).w;
        return float4(a>0?c/max(sum,1e-8f):float3(0),a);
    }
    float2 dir=0;float len=0;
    fsr_set(dir,len,(1-pp.x)*(1-pp.y),l[0],l[5],l[4],l[9],l[3]);
    fsr_set(dir,len,pp.x*(1-pp.y),l[1],l[4],l[9],l[8],l[6]);
    fsr_set(dir,len,(1-pp.x)*pp.y,l[4],l[2],l[3],l[6],l[11]);
    fsr_set(dir,len,pp.x*pp.y,l[9],l[3],l[6],l[7],l[10]);
    float length2=dot(dir,dir);dir=length2<1.f/32768.f?float2(1,0):dir/sqrt(length2);
    len*=.5f;len*=len;float stretch=dot(dir,dir)/max(abs(dir.x),abs(dir.y));
    float2 lengths=float2(1+(stretch-1)*len,1-.5f*len);
    float lobe=.5f+(.21f-.5f)*len,clip=1/lobe,total=0;float3 color=0;
    for(uint k=0;k<12;k++){
        float2 off=float2(offsets[k])-pp;
        float2 v=float2(dot(off,dir),dot(off,float2(-dir.y,dir.x)))*lengths;
        float d2=min(dot(v,v),clip),wb=.4f*d2-1,wa=lobe*d2-1;
        wb=1.5625f*wb*wb-.5625f;float weight=wb*wa*wa;color+=taps[k].xyz*weight;total+=weight;
    }
    float3 lo=min(min(taps[4].xyz,taps[9].xyz),min(taps[3].xyz,taps[6].xyz));
    float3 hi=max(max(taps[4].xyz,taps[9].xyz),max(taps[3].xyz,taps[6].xyz));
    return float4(clamp(color/total,lo,hi),1);
}
kernel void fsr_easu(device const uchar *src [[buffer(0)]],device uchar *dst [[buffer(1)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint w=p.u[0],h=p.u[1],ow=p.u[2],oh=p.u[3],channels=p.u[4];if(i>=ow*oh)return;
    float2 xy=(float2(i%ow,i/ow)+.5f)*float2(w,h)/float2(ow,oh)-.5f;
    fsr_store(dst,i,channels,fsr_easu_sample(src,xy,w,h,channels));
}
kernel void fsr_rcas(device const uchar *src [[buffer(0)]],device uchar *dst [[buffer(1)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint w=p.u[0],h=p.u[1],channels=p.u[2];if(i>=w*h)return;int2 xy=int2(i%w,i/w);
    float4 b4=fsr_load(src,xy+int2(0,-1),w,h,channels),d4=fsr_load(src,xy+int2(-1,0),w,h,channels);
    float4 e4=fsr_load(src,xy,w,h,channels),f4=fsr_load(src,xy+int2(1,0),w,h,channels),h4=fsr_load(src,xy+int2(0,1),w,h,channels);
    if(min(min(b4.w,d4.w),min(e4.w,min(f4.w,h4.w)))<1.f){fsr_store(dst,i,channels,e4);return;}
    float3 b=b4.xyz,d=d4.xyz,e=e4.xyz,f=f4.xyz,v=h4.xyz;
    float bL=fsr_luma(b),dL=fsr_luma(d),eL=fsr_luma(e),fL=fsr_luma(f),hL=fsr_luma(v);
    float range=max(max(bL,dL),max(eL,max(fL,hL)))-min(min(bL,dL),min(eL,min(fL,hL)));
    float noise=1-.5f*clamp(abs(.25f*(bL+dL+fL+hL)-eL)/max(range,1e-8f),0.f,1.f);
    float3 mn=min(min(b,d),min(f,v)),mx=max(max(b,d),max(f,v));
    float3 hitMin=min(mn,e)/max(4*mx,float3(1e-8f));
    float3 hitMax=(1-max(mx,e))/min(4*mn-4,float3(-1e-8f));
    float3 lobes=max(-hitMin,hitMax);float lobe=max(-.1875f,min(max(lobes.x,max(lobes.y,lobes.z)),0.f))*p.f[0]*noise;
    fsr_store(dst,i,channels,float4((lobe*(b+d+v+f)+e)/(4*lobe+1),e4.w));
}

// ADM experiment, NOT AMD FSR3 frame generation. Planar reprojection supplies
// motion; unobserved pixels stay transparent, and disagreement can veto blends.
inline float2 photo_project(float2 xy,constant float *h) {
    float d=h[6]*xy.x+h[7]*xy.y+h[8];
    return float2(h[0]*xy.x+h[1]*xy.y+h[2],h[3]*xy.x+h[4]*xy.y+h[5])/d;
}
kernel void photo_interpolate(device const uchar *a [[buffer(0)]],device const uchar *b [[buffer(1)]],device uchar *dst [[buffer(2)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint w=p.u[0],h=p.u[1];if(i>=w*h)return;float2 xy=float2(i%w,i/w);
    float2 qa=photo_project(xy,p.f),qb=photo_project(xy,p.f+9);
    bool va=all(isfinite(qa))&&all(qa>=0)&&all(qa<=float2(w-1,h-1));
    bool vb=all(isfinite(qb))&&all(qb>=0)&&all(qb<=float2(w-1,h-1));
    float4 ca=va?fsr_bilinear(a,qa,w,h,4):float4(0),cb=vb?fsr_bilinear(b,qb,w,h,4):float4(0);
    va=va&&ca.w>.999f;vb=vb&&cb.w>.999f;float4 value=0;
    if(va&&vb){
        float3 delta=abs(ca.xyz-cb.xyz);
        bool agree=max(delta.x,max(delta.y,delta.z))<=p.f[19];
        value=agree||!p.u[2]?mix(ca,cb,p.f[18]):(p.f[18]<=.5f?ca:cb);
    }else if(va)value=ca;else if(vb)value=cb;
    fsr_store(dst,i,4,value);
}
// Generated colors are eligible only inside shared real-photo support.
kernel void fsr_validate_generated(device const uchar *previous [[buffer(0)]],device const uchar *current [[buffer(1)]],device uchar4 *output [[buffer(2)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint w=p.u[0],h=p.u[1];if(i>=w*h)return;float2 point=float2(i%w,i/w);float4 samples[2];bool valid=true;
    for(uint n=0;n<2;n++){
        uint o=n*9;float z=p.f[o+6]*point.x+p.f[o+7]*point.y+p.f[o+8];
        float2 q=float2(p.f[o]*point.x+p.f[o+1]*point.y+p.f[o+2],p.f[o+3]*point.x+p.f[o+4]*point.y+p.f[o+5])/z;
        if(!all(isfinite(q))||any(q<0)||q.x>w-1||q.y>h-1){valid=false;samples[n]=float4(0);continue;}
        samples[n]=fsr_bilinear(n?current:previous,q,w,h,p.u[2+n]);valid=valid&&samples[n].w>.999f;
    }
    float3 a=samples[0].xyz,b=samples[1].xyz,c=float3(output[i].xyz)/255.f;
    valid=valid&&all(abs(a-b)<=48.f/255.f)&&all(c>=min(a,b)-20.f/255.f)&&all(c<=max(a,b)+20.f/255.f);
    output[i]=valid?uchar4(output[i].xyz,255):uchar4(0);
}
