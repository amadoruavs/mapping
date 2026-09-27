#include <metal_stdlib>
using namespace metal;
struct Params { uint u[16]; float f[32]; };

inline int reflect101(int x,int n) {
    if(n<=1) return 0;
    int period=2*n-2; x=((x%period)+period)%period; return min(x,period-x);
}
inline float3 pixel(device const packed_uchar3 *im,int x,int y,uint w,uint h) {
    return x<0||y<0||x>=int(w)||y>=int(h) ? float3(0) : float3(im[y*w+x]);
}
inline float edge(int x,int y,uint w,uint h) {
    if(x<0||y<0||x>=int(w)||y>=int(h)) return 0;
    float d=min(min(x/(w*.5f),(w-1-x)/(w*.5f)),min(y/(h*.5f),(h-1-y)/(h*.5f)));
    return d<.02f ? 0 : clamp(d*8,0.f,1.f);
}
inline float2 transform(float2 xy,constant Params &p) {
    float z=p.f[6]*xy.x+p.f[7]*xy.y+p.f[8];
    return float2(p.f[0]*xy.x+p.f[1]*xy.y+p.f[2],p.f[3]*xy.x+p.f[4]*xy.y+p.f[5])/z;
}
// OpenCV 4 uses a 5-bit table; OpenCV 5 uses full bilinear precision.
inline float2 interpolation_position(float2 xy,bool full){return full?xy:rint(xy*32.f)/32.f;}
inline float3 interpolate(device const packed_uchar3 *im,float2 xy,uint w,uint h,bool full) {
    xy=interpolation_position(xy,full);int2 base=int2(floor(xy));float2 f=xy-float2(base);
    float3 a=mix(pixel(im,base.x,base.y,w,h),pixel(im,base.x+1,base.y,w,h),f.x);
    float3 b=mix(pixel(im,base.x,base.y+1,w,h),pixel(im,base.x+1,base.y+1,w,h),f.x);
    return full?rint(mix(a,b,f.y)):floor(mix(a,b,f.y)+.5f);
}
kernel void warp_confidence(device const packed_uchar3 *im [[buffer(0)]],device packed_uchar3 *rgb [[buffer(1)]],device float *weight [[buffer(2)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint w=p.u[0],h=p.u[1],ow=p.u[2],oh=p.u[3]; if(i>=ow*oh) return;
    float2 xy=transform(float2(i%ow,i/ow),p);
    if(!all(isfinite(xy))||any(abs(xy)>float2(1e7))) { rgb[i]=packed_uchar3(0);weight[i]=0;return; }
    rgb[i]=packed_uchar3(uchar3(interpolate(im,xy,w,h,p.u[15])));
    xy=interpolation_position(xy,p.u[15]);int2 b=int2(floor(xy));float2 f=xy-float2(b);
    weight[i]=mix(mix(edge(b.x,b.y,w,h),edge(b.x+1,b.y,w,h),f.x),mix(edge(b.x,b.y+1,w,h),edge(b.x+1,b.y+1,w,h),f.x),f.y);
}
kernel void rectify(device const packed_uchar3 *im [[buffer(0)]],device packed_uchar3 *rgb [[buffer(1)]],device uchar *valid [[buffer(2)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint w=p.u[0],h=p.u[1],ow=p.u[2],oh=p.u[3];if(i>=ow*oh)return;
    float2 xy=transform(float2(i%ow,i/ow)+float2(p.f[9],p.f[10]),p);
    bool finite=all(isfinite(xy))&&all(abs(xy)<float2(1e7));
    rgb[i]=finite?packed_uchar3(uchar3(interpolate(im,xy,w,h,p.u[15]))):packed_uchar3(0);
    valid[i]=finite&&all(xy>=0)&&xy.x<w-1&&xy.y<h-1;
}
// One 128-thread group per NCC candidate. Cooperative sums avoid im2col storage.
kernel void ncc(device const packed_uchar3 *im [[buffer(0)]],device const packed_uchar3 *patch [[buffer(1)]],device const uchar *valid [[buffer(2)]],device float *scores [[buffer(3)]],constant Params &p [[buffer(15)]],uint g [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    uint s=p.u[0],w=p.u[1],ow=w-s+1; if(g>=ow*ow)return;
    float3 a=0,b=0;float aa=0,bb=0,ab=0;uint bad=0;
    for(uint k=t;k<s*s;k+=128) {
        uint q=(g/ow+k/s)*w+g%ow+k%s;float3 x=float3(im[q]),y=float3(patch[k]);
        a+=x;b+=y;aa+=dot(x,x);bb+=dot(y,y);ab+=dot(x,y);bad+=!valid[q];
    }
    a=simd_sum(a);b=simd_sum(b);aa=simd_sum(aa);bb=simd_sum(bb);ab=simd_sum(ab);bad=simd_sum(bad);
    threadgroup float4 sa[4],sb[4],sc[4];
    if(lane==0){sa[sg]=float4(a,aa);sb[sg]=float4(b,bb);sc[sg]=float4(ab,bad,0,0);}threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t==0){float4 x=0,y=0,z=0;for(uint k=0;k<4;k++){x+=sa[k];y+=sb[k];z+=sc[k];}
        float count=s*s,va=max(x.w-dot(x.xyz,x.xyz)/count,0.f),vb=max(y.w-dot(y.xyz,y.xyz)/count,0.f);
        scores[g]=z.y>0?-1.f:vb<1e-6f?1.f:va<1e-6f?0.f:clamp((z.x-dot(x.xyz,y.xyz)/count)/sqrt(va*vb),-1.f,1.f);
    }
}
kernel void descriptor_norm(device const float *d [[buffer(0)]],device float *norm [[buffer(1)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.u[0])return;float v=0;for(uint k=0;k<p.u[1];k++)v+=d[i*p.u[1]+k]*d[i*p.u[1]+k];norm[i]=v;
}
// One SIMD group per row. Exact top-two L2 after the MPS dot product, stable ties.
kernel void nearest_two(device const float *dotp [[buffer(0)]],device const float *an [[buffer(1)]],device const float *bn [[buffer(2)]],device float4 *out [[buffer(3)]],constant Params &p [[buffer(15)]],uint g [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]]) {
    if(g>=p.u[0])return;float first=INFINITY,second=INFINITY;uint fi=0xffffffff,si=0xffffffff;
    for(uint j=t;j<p.u[1];j+=32){float d=max(an[g+p.u[2]]+bn[j]-2*dotp[g*p.u[1]+j],0.f);
        if(d<first||(d==first&&j<fi)){second=first;si=fi;first=d;fi=j;}else if(d<second||(d==second&&j<si)){second=d;si=j;}}
    float best=simd_min(first);uint index=simd_min(first==best?fi:0xffffffff);
    float next=fi==index?second:first;uint ni=fi==index?si:fi;float runner=simd_min(next);uint runnerIndex=simd_min(next==runner?ni:0xffffffff);
    if(t==0)out[g+p.u[2]]=float4(index,runnerIndex,best,runner);
}
kernel void pack_tiles(device const packed_uchar3 *im [[buffer(0)]],device packed_uchar3 *tiles [[buffer(1)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint w=p.u[0],h=p.u[1],s=p.u[2],cols=(w+s-1)/s;if(i>=p.u[3])return;
    uint tile=i/(s*s),k=i%(s*s),x=(tile%cols)*s+k%s,y=(tile/cols)*s+k/s;
    tiles[i]=x<w&&y<h?im[y*w+x].zyx:packed_uchar3(0);
}
kernel void accumulate(device const packed_uchar3 *im [[buffer(0)]],device const float *confidence [[buffer(1)]],device float4 *sum [[buffer(2)]],device ushort *coverage [[buffer(3)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.u[0]*p.u[1])return;uint j=(i/p.u[0]+p.u[3])*p.u[4]+i%p.u[0]+p.u[2];float c=confidence[i];sum[j]+=float4(float3(im[i])*c,c);coverage[j]+=ushort(c>.001f);
}
kernel void confidence_half(device const float *src [[buffer(0)]],device half *dst [[buffer(1)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i<p.u[0])dst[i]=half(src[i]);
}
kernel void finish_mean(device float4 *sum [[buffer(0)]],device float *best [[buffer(1)]],device uchar4 *selected [[buffer(2)]],device ushort *ids [[buffer(3)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.u[0])return;float4 s=sum[i];sum[i]=float4(s.xyz/max(s.w,1e-6f),s.w);best[i]=-INFINITY;selected[i]=0;ids[i]=65535;
}
inline float normal_cdf(float x) {
    // Abramowitz-Stegun, absolute error < 7.5e-8 for the normal CDF.
    float t=1.f/(1.f+.2316419f*abs(x));float a=t*(.319381530f+t*(-.356563782f+t*(1.781477937f+t*(-1.821255978f+t*1.330274429f))));
    float v=exp(-.5f*x*x)*.3989422804014327f*a;return x>=0?1.f-v:v;
}
kernel void select_source(device const packed_uchar3 *im [[buffer(0)]],device const half *confidence [[buffer(1)]],device const float4 *mean [[buffer(2)]],device const float *coarse [[buffer(3)]],device float *best [[buffer(4)]],device uchar4 *selected [[buffer(5)]],device ushort *ids [[buffer(6)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint rw=p.u[0],rh=p.u[1],w=p.u[4],h=p.u[5],gw=p.u[6],gh=p.u[7];if(i>=rw*rh)return;
    uint x=i%rw+p.u[2],y=i/rw+p.u[3],j=y*w+x;float c=confidence[i];if(c<=.001f)return;
    float2 q=clamp((float2(x,y)+.5f)*float2(gw,gh)/float2(w,h)-.5f,float2(0),float2(gw-1,gh-1));uint2 b=uint2(floor(q)),n=min(b+1,uint2(gw-1,gh-1));float2 f=fract(q);
    float field=mix(mix(coarse[b.y*gw+b.x],coarse[b.y*gw+n.x],f.x),mix(coarse[n.y*gw+b.x],coarse[n.y*gw+n.x],f.x),f.y);
    float2 variance=(1.f-f)*(1.f-f)+f*f;float u=clamp(normal_cdf(field/sqrt(variance.x*variance.y)),.00001f,.99999f);
    float3 delta=float3(im[i])-mean[j].xyz;float score=-dot(delta,delta)/(6*p.f[0]*p.f[0])+log(max(c,1e-8f))-log(-log(u));
    if(score>best[j]){best[j]=score;selected[j]=uchar4(uchar3(im[i]),255);ids[j]=p.u[8];}
}
kernel void seam_input(device const float4 *mean [[buffer(0)]],device const uchar4 *selected [[buffer(1)]],device float4 *diff [[buffer(2)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.u[0])return;bool valid=mean[i].w>.001f;diff[i]=valid?float4(mean[i].xyz-float3(selected[i].xyz),1):float4(0);
}
kernel void seam_output(device const float4 *blur [[buffer(0)]],device const float4 *mean [[buffer(1)]],device uchar4 *selected [[buffer(2)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.u[0])return;float4 b=blur[i];selected[i]=mean[i].w>.001f?uchar4(uchar3(clamp(float3(selected[i].xyz)+b.xyz/max(b.w,1e-6f),0.f,255.f)),255):uchar4(0);
}
kernel void pad_texture(device const float4 *src [[buffer(0)]],texture2d<float,access::write> dst [[texture(0)]],constant Params &p [[buffer(15)]],uint2 q [[thread_position_in_grid]]) {
    if(q.x>=p.u[7]||q.y>=p.u[8])return;
    int x=reflect101(int(q.x+p.u[2])-int(p.u[6]),p.u[0]),y=reflect101(int(q.y+p.u[3])-int(p.u[6]),p.u[1]);dst.write(src[y*p.u[0]+x],q);
}
kernel void crop_texture(device float4 *dst [[buffer(0)]],texture2d<float,access::read> src [[texture(0)]],constant Params &p [[buffer(15)]],uint2 q [[thread_position_in_grid]]) {
    uint2 pos=q+uint2(p.u[2],p.u[3]);if(q.x>=p.u[4]||q.y>=p.u[5]||pos.x>=p.u[0]||pos.y>=p.u[1])return;dst[pos.y*p.u[0]+pos.x]=src.read(q+p.u[6]);
}
inline float3 decode(float3 c){return select(pow((c+.055f)/1.055f,float3(2.4f)),c/12.92f,c<=.04045f);}
inline float3 encode(float3 c){return select(1.055f*pow(max(c,0.f),float3(1.f/2.4f))-.055f,12.92f*c,c<=.0031308f);}
inline float3 shade(float3 rgb,float3 normal,float vis,float3 sun,float strength){
    float direct=max(dot(normal,sun),0.f)/max(sun.z,.15f)*vis;
    float s=1+strength*(clamp(.65f+.35f*direct,.35f,1.45f)-1);
    return rgb*float3(1.065f,1.015f,.94f)*s+float3(.008f,.004f,0)*(1-dot(rgb,float3(.2126f,.7152f,.0722f)));
}
kernel void shade_reference(device const packed_float3 *rgb [[buffer(0)]],device const packed_float3 *normals [[buffer(1)]],device const float *visibility [[buffer(2)]],device packed_float3 *out [[buffer(3)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i<p.u[0])out[i]=packed_float3(shade(float3(rgb[i]),float3(normals[i]),visibility[i],float3(p.f[0],p.f[1],p.f[2]),p.f[3]));
}
kernel void flat_light(device const uchar4 *rgba [[buffer(0)]],device packed_float3 *linear [[buffer(1)]],device uchar4 *encoded [[buffer(2)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.u[0])return;uchar4 c=rgba[i];float3 rgb=decode(float3(c.zyx)/255.f);
    float3 v=c.w?shade(rgb,float3(0,0,1),1,float3(p.f[0],p.f[1],p.f[2]),p.f[3]):float3(0);
    linear[i]=packed_float3(v);encoded[i]=uchar4(uchar3(rint(clamp(encode(v.zyx),0.f,1.f)*255.f)),c.w);
}
kernel void terrain_normals(device const float *height [[buffer(0)]],device float4 *normal [[buffer(1)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint w=p.u[0],h=p.u[1];if(i>=w*p.u[3])return;uint x=i%w,y=i/w+p.u[2];
    uint xl=x?x-1:0,xr=min(x+1,w-1),yt=y?y-1:0,yb=min(y+1,h-1);
    float dx=(height[y*w+xr]-height[y*w+xl])/(max(xr-xl,1u)*p.f[0]);float dy=(height[yb*w+x]-height[yt*w+x])/(max(yb-yt,1u)*p.f[0]);normal[i]=float4(normalize(float3(-dx,dy,1)),0);
}
inline float height_pixel(device const float *height,int x,int y,uint w,uint h){return x<0||y<0||x>=int(w)||y>=int(h)?-1e6f:height[y*w+x];}
kernel void terrain_visibility(device const float *height [[buffer(0)]],device const uchar4 *rgba [[buffer(1)]],device float4 *visibility [[buffer(2)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint w=p.u[0],h=p.u[1];if(i>=w*p.u[3])return;uint x=i%w,y=reflect101(int(i/w)+int(p.u[2]),h);float base=height[y*w+x],v=1;
    for(uint k=0;k<p.u[4];k++){
        float d=p.f[0]+k*p.f[1];float2 xy=float2(x,y)+float2(p.f[2],p.f[3])*d;
        int2 nearest=int2(rint(xy));if(nearest.x<0||nearest.y<0||nearest.x>=int(w)||nearest.y>=int(h)||!rgba[nearest.y*w+nearest.x].w)continue;
        xy=interpolation_position(xy,p.u[15]);int2 b=int2(floor(xy));float2 f=xy-float2(b);
        float z=mix(mix(height_pixel(height,b.x,b.y,w,h),height_pixel(height,b.x+1,b.y,w,h),f.x),mix(height_pixel(height,b.x,b.y+1,w,h),height_pixel(height,b.x+1,b.y+1,w,h),f.x),f.y);
        if(z>base+d*p.f[4]+.05f){v=0;break;}
    }visibility[i]=float4(v);
}
kernel void light_candidate(device const uchar4 *rgba [[buffer(0)]],device const float4 *normal [[buffer(1)]],device const float4 *visibility [[buffer(2)]],device float4 *candidate [[buffer(3)]],device float4 *mean [[buffer(4)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.u[0])return;float3 rgb=decode(float3(rgba[i+p.u[1]].zyx)/255.f);float3 v=shade(rgb,normal[i].xyz,visibility[i+p.u[2]].x,float3(p.f[0],p.f[1],p.f[2]),p.f[3]);candidate[i]=float4(v,0);mean[i]+=float4(v*p.f[4],0);
}
kernel void choose_light(device const float4 *candidate [[buffer(0)]],device const float4 *mean [[buffer(1)]],device const float *uniform [[buffer(2)]],device float *best [[buffer(3)]],device float4 *chosen [[buffer(4)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.u[0])return;float3 d=candidate[i].xyz-mean[i].xyz;float score=-dot(d,d)/(6*.025f*.025f)-log(-log(uniform[i]));if(p.u[1]==0||score>best[i]){best[i]=score;chosen[i]=candidate[i];}
}
kernel void export_light(device const float4 *chosen [[buffer(0)]],device const uchar4 *rgba [[buffer(1)]],device packed_float3 *linear [[buffer(2)]],device uchar4 *encoded [[buffer(3)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.u[0])return;uint j=i+p.u[1];uchar a=rgba[j].w;float3 v=a?chosen[i].xyz:float3(0);linear[j]=packed_float3(v);encoded[j]=uchar4(uchar3(rint(clamp(encode(v.zyx),0.f,1.f)*255.f)),a);
}
