// ISM: consolidated ADM + AMD FSR shader source.
// Host/file operations are in runtime/ and run.py. See GUIDE.md.
#include <metal_stdlib>
#include <metal_atomic>
#include <metal_raytracing>
#include <simd/simd.h>

// ===== ADM pipeline =====
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

// ===== ADM alignment =====
// Device-resident robust projective alignment. Eight parameters per image.
inline float2 align_project(device const float *x,uint id,float2 p) {
    device const float *h=x+id*8;float z=h[6]*p.x+h[7]*p.y+1;
    return float2(h[0]*p.x+h[1]*p.y+h[2],h[3]*p.x+h[4]*p.y+h[5])/z;
}
inline float align_derivative(uint k,uint axis,float2 p,float2 q,float invz) {
    if(k<6)return (k/3==axis)?(k%3==0?p.x:k%3==1?p.y:1.f)*invz:0.f;
    return -q[axis]*(k==6?p.x:p.y)*invz;
}
kernel void align_linearize(device const float *x [[buffer(0)]],device const float4 *points [[buffer(1)]],device const uint2 *owners [[buffer(2)]],device float *jac [[buffer(3)]],device float4 *rw [[buffer(4)]],device float *cost [[buffer(5)]],constant Params &p [[buffer(15)]],uint k [[thread_position_in_grid]]) {
    if(k>=p.u[0])return;uint2 ids=owners[k];float4 xy=points[k]/p.f[0];
    float2 a=align_project(x,ids.x,xy.xy),b=align_project(x,ids.y,xy.zw),r=a-b;
    float2 iz=1.f/float2(x[ids.x*8+6]*xy.x+x[ids.x*8+7]*xy.y+1,x[ids.y*8+6]*xy.z+x[ids.y*8+7]*xy.w+1);
    float2 pixel=r*p.f[0],weights=rsqrt(1+pixel*pixel);rw[k]=float4(r,weights);
    float2 squared=pixel*pixel;cost[k]=dot(squared/(sqrt(1+squared)+1),float2(1));
    for(uint axis=0;axis<2;axis++)for(uint j=0;j<8;j++){
        jac[k*32+axis*16+j]=align_derivative(j,axis,xy.xy,a,iz.x);
        jac[k*32+axis*16+j+8]=-align_derivative(j,axis,xy.zw,b,iz.y);
    }
}
kernel void align_edge_blocks(device const float *jac [[buffer(0)]],device const float4 *rw [[buffer(1)]],device const uint4 *edges [[buffer(2)]],device float *blocks [[buffer(3)]],constant Params &p [[buffer(15)]],uint k [[thread_position_in_grid]]) {
    if(k>=p.u[0]*272)return;uint e=k/272,q=k%272;uint4 info=edges[e];float value=0;
    uint row=q<256?q/16:q-256,col=q%16;
    for(uint o=info.z;o<info.z+info.w;o++){
        float4 r=rw[o];float2 a=float2(jac[o*32+row],jac[o*32+16+row]);
        float2 b=q<256?float2(jac[o*32+col],jac[o*32+16+col]):r.xy;
        value+=dot(a*b,r.zw);
    }blocks[k]=value;
}
kernel void align_assemble(device const float *blocks [[buffer(0)]],device const uint4 *edges [[buffer(1)]],device const int *links [[buffer(2)]],device const uint *offsets [[buffer(3)]],device const uint2 *incidence [[buffer(4)]],device float *normal [[buffer(5)]],device float *rhs [[buffer(6)]],device const float *state [[buffer(7)]],constant Params &p [[buffer(15)]],uint k [[thread_position_in_grid]]) {
    uint n=p.u[0],d=n*8;if(k>=d*d+d)return;bool grad=k>=d*d;uint row=grad?k-d*d:k/d,col=grad?row:k%d;
    uint i=row/8,j=col/8,r=row%8,c=col%8;float value=0;
    if(i==p.u[1]||j==p.u[1])value=(!grad&&row==col)?1:0;
    else if(i==j){
        for(uint a=offsets[i];a<offsets[i+1];a++){
            uint2 v=incidence[a];uint base=v.y*8;
            value+=blocks[v.x*272+(grad?256+base+r:(base+r)*16+base+c)];
        }
        if(!grad&&row==col)value+=state[0]*max(value,1e-8f)+1e-9f;
    }else{int e=links[i*n+j];if(e>=0){uint4 info=edges[e];uint rb=i==info.x?0:8,cb=j==info.x?0:8;value=blocks[e*272+(rb+r)*16+cb+c];}}
    if(grad)rhs[row]=-value;else normal[k]=value;
}
kernel void align_trial(device const float *x [[buffer(0)]],device const float *delta [[buffer(1)]],device float *trial [[buffer(2)]],device const int *status [[buffer(3)]],device const float *state [[buffer(4)]],constant Params &p [[buffer(15)]],uint k [[thread_position_in_grid]]) {
    if(k<p.u[0])trial[k]=status[0]==0&&state[6]==0?x[k]+delta[k]:x[k];
}
kernel void align_cost(device const float *x [[buffer(0)]],device const float4 *points [[buffer(1)]],device const uint2 *owners [[buffer(2)]],device float *cost [[buffer(3)]],constant Params &p [[buffer(15)]],uint k [[thread_position_in_grid]]) {
    if(k>=p.u[0])return;float4 xy=points[k]/p.f[0];uint2 ids=owners[k];
    float za=x[ids.x*8+6]*xy.x+x[ids.x*8+7]*xy.y+1,zb=x[ids.y*8+6]*xy.z+x[ids.y*8+7]*xy.w+1;
    float2 r=(align_project(x,ids.x,xy.xy)-align_project(x,ids.y,xy.zw))*p.f[0];float2 squared=r*r;
    cost[k]=za>.05f&&zb>.05f&&all(isfinite(r))?dot(squared/(sqrt(1+squared)+1),float2(1)):INFINITY;
}
kernel void align_accept(device const float *oldCost [[buffer(0)]],device const float *newCost [[buffer(1)]],device const int *status [[buffer(2)]],device float *state [[buffer(3)]],device const float *delta [[buffer(4)]],device const float *x [[buffer(5)]],device const float *rhs [[buffer(6)]],constant Params &p [[buffer(15)]],uint k [[thread_position_in_grid]]) {
    if(k)return;state[1]=0;if(state[6])return;
    float old=oldCost[0],next=newCost[0],damping=state[0];if(p.u[0]==0)state[7]=old;
    if(status[0]==0&&isfinite(next)&&next<old){
        state[1]=1;state[2]++;state[0]=max(state[0]*.35f,1e-7f);state[4]=next;
        // Bidirectional transfer includes inverse/projection cancellation in
        // float32, so its relative objective floor is 1e-5 (legacy: 1e-6).
        if((!p.u[2]||damping<.01f)&&old-next<(p.u[2]?1e-5f:1e-6f)*max(old,1.f))state[6]=1;
    }else{state[3]++;state[0]=min(state[0]*8.f,1e12f);state[4]=old;}
    float step=0,norm=0;for(uint i=0;i<p.u[1];i++){step+=delta[i]*delta[i];norm+=x[i]*x[i];}
    uint trace=16+p.u[0]*4;state[trace]=old;state[trace+1]=next;state[trace+2]=state[0];state[trace+3]=sqrt(step/max(norm,1.f));
    // About 2.5 float32 ulps: smaller normalized updates oscillate in the
    // projection arithmetic. Low damping is required to exclude artificial stalls.
    if(status[0]==0&&(state[2]>0||old<=1e-8f)&&state[0]<.01f&&step<=9e-14f*max(norm,1.f))state[6]=2;
    // Near a float32 minimum the rounded cost can increase by a few ulps.
    // A tiny predicted first-order gain with LOW damping establishes local
    // stationarity; a tiny heavily damped step alone never establishes it.
    if(p.u[2]&&status[0]==0&&damping<.01f&&state[2]>0&&isfinite(next)){
        float predicted=0;for(uint i=0;i<p.u[1];i++)predicted+=delta[i]*rhs[i];
        predicted*=p.f[0]*p.f[0];
        if(predicted>=0&&predicted<=1e-5f*max(old,1.f)&&abs(next-old)<=1e-4f*max(old,1.f))state[6]=3;
    }
    state[5]=p.u[0]+1;state[8]=status[0];
}
kernel void align_commit(device float *x [[buffer(0)]],device const float *trial [[buffer(1)]],device const float *state [[buffer(2)]],constant Params &p [[buffer(15)]],uint k [[thread_position_in_grid]]) {
    if(k<p.u[0]&&state[1])x[k]=trial[k];
}

// Measure a match in the receiving photograph's pixels. Unlike distances in
// the mosaic, these residuals cannot improve by shrinking the mosaic cameras.
// Each match occupies two consecutive rows: i->j, then j->i, each weight 1/2.
inline float3x3 align_matrix(device const float *x,uint id) {
    device const float *h=x+id*8;
    return float3x3(float3(h[0],h[3],h[6]),float3(h[1],h[4],h[7]),float3(h[2],h[5],1));
}
inline float3x3 align_inverse(float3x3 H,thread float &det) {
    float3x3 rows=transpose(H);
    float3 a=cross(rows[1],rows[2]),b=cross(rows[2],rows[0]),c=cross(rows[0],rows[1]);
    det=dot(rows[0],a);
    return float3x3(a,b,c)/det;
}
kernel void align_transfer_linearize(device const float *x [[buffer(0)]],device const float4 *points [[buffer(1)]],device const uint2 *owners [[buffer(2)]],device float *jac [[buffer(3)]],device float4 *rw [[buffer(4)]],device float *cost [[buffer(5)]],constant Params &p [[buffer(15)]],uint k [[thread_position_in_grid]]) {
    if(k>=p.u[0])return;bool reverse=k&1;uint2 ids=owners[k];float4 xy=points[k]/p.f[0];
    if(reverse){ids=ids.yx;xy=xy.zwxy;}
    float det;float3x3 inverse=align_inverse(align_matrix(x,ids.y),det);
    float3 a=float3(xy.xy,1),u=inverse*(align_matrix(x,ids.x)*a);float2 q=u.xy/u.z;
    float2 r=q-xy.zw,pixel=r*p.f[0],sq=pixel*pixel;
    rw[k]=float4(r,.5f*rsqrt(1+sq));cost[k]=.5f*dot(sq/(sqrt(1+sq)+1),float2(1));
    float3 qh=float3(q,1);
    for(uint axis=0;axis<2;axis++)for(uint j=0;j<8;j++) {
        uint row=j/3,col=j%3;
        float factor=inverse[row][axis]-q[axis]*inverse[row][2];
        jac[k*32+axis*16+j+(reverse?8:0)]=factor*a[col]/u.z;
        jac[k*32+axis*16+j+(reverse?0:8)]=-factor*qh[col];
    }
}
kernel void align_transfer_cost(device const float *x [[buffer(0)]],device const float4 *points [[buffer(1)]],device const uint2 *owners [[buffer(2)]],device float *cost [[buffer(3)]],constant Params &p [[buffer(15)]],uint k [[thread_position_in_grid]]) {
    if(k>=p.u[0])return;uint2 ids=owners[k];float4 xy=points[k]/p.f[0];
    if(k&1){ids=ids.yx;xy=xy.zwxy;}
    float det;float3x3 source=align_matrix(x,ids.x),target=align_matrix(x,ids.y);
    float3x3 inverse=align_inverse(target,det);float3 v=source*float3(xy.xy,1),u=inverse*v;
    float2 r=(u.xy/u.z-xy.zw)*p.f[0],sq=r*r;
    float z=(target*float3(xy.zw,1)).z;
    bool valid=abs(det)>1e-12f&&v.z>.05f&&z>.05f&&u.z>1e-6f&&all(isfinite(r));
    cost[k]=valid?.5f*dot(sq/(sqrt(1+sq)+1),float2(1)):INFINITY;
}

// ===== ADM geographic =====
// OpenSfM geographic constraints adapted for a fixed ground plane.
// Upstream: OpenDroneMap/OpenSfM c5328439465e6ace011f39077d1077d7b1cdd65d
// DataPriorError (position) and UpVectorError with Cauchy loss (orientation).
// See runtime/OPENSFM_PORT.md and LICENSES.txt in the packaged ISM.
// Two observed camera axes constrain full orientation, including nadir yaw.
// Six physical pose parameters occupy eight slots for the shared block solver.
inline float3x3 geo_load(device const float *a) {
    return float3x3(float3(a[0],a[1],a[2]),float3(a[3],a[4],a[5]),float3(a[6],a[7],a[8]));
}
inline void geo_store(device float *a,float3x3 m) {
    for(uint c=0;c<3;c++)for(uint r=0;r<3;r++)a[c*3+r]=m[c][r];
}
inline float3x3 geo_cross(float3 a) {
    return float3x3(float3(0,a.z,-a.y),float3(-a.z,0,a.x),float3(a.y,-a.x,0));
}
inline float3x3 geo_exp(float3 a,thread float3x3 &left) {
    float t2=dot(a,a),A,B,C;
    if(t2<1e-4f){A=1-t2/6+t2*t2/120;B=.5f-t2/24+t2*t2/720;C=1.f/6-t2/120+t2*t2/5040;}
    else{float t=sqrt(t2);A=sin(t)/t;B=(1-cos(t))/t2;C=(1-A)/t2;}
    float3x3 K=geo_cross(a),K2=K*K;left=float3x3(1)+B*K+C*K2;
    return float3x3(1)+A*K+B*K2;
}
inline float3x3 geo_ground(float3 C,float3x3 rays) {
    float3x3 rows=transpose(rays);
    return transpose(float3x3(C.z*rows[0]-C.x*rows[2],C.z*rows[1]-C.y*rows[2],-rows[2]));
}
kernel void geo_camera_model(device const float *x [[buffer(0)]],device const float *data [[buffer(1)]],
    device float *matrices [[buffer(2)]],device float *derivatives [[buffer(3)]],
    device float *priorBlocks [[buffer(4)]],device float *cost [[buffer(5)]],
    constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.u[0])return;device const float *b=data+i*32,*v=x+i*8;
    float3 C=float3(v[0],v[1],v[2]),a=float3(v[3],v[4],v[5]);
    float3x3 left,E=geo_exp(a,left),base=geo_load(b),R=E*base,rays=R*geo_load(b+9);
    float3x3 H=geo_ground(C,rays);geo_store(matrices+i*9,H);
    float3x3 dH[8];
    for(uint k=0;k<8;k++) {
        if(k<3){float3 e=0;e[k]=1;float3x3 rows=transpose(rays);
            dH[k]=transpose(float3x3(e.z*rows[0]-e.x*rows[2],e.z*rows[1]-e.y*rows[2],float3(0)));}
        else if(k<6)dH[k]=geo_ground(C,geo_cross(left[k-3])*rays);
        else dH[k]=float3x3(0);
        geo_store(derivatives+(i*8+k)*9,dH[k]);
    }
    // This is OpenSfM's position DataPriorError: (position-prior)/sigma,
    // with fixed identity GPS bias and separate configured AGL uncertainty.
    float residual[9],J[9][8],weight[9];
    float invwidth=1/p.f[0];
    for(uint r=0;r<9;r++)for(uint k=0;k<8;k++)J[r][k]=0;
    float3 gps=(C-float3(b[18],b[19],b[20]))*float3(b[21],b[22],b[23]);
    cost[p.u[1]+i*3]=.5f*dot(gps,gps);
    for(uint r=0;r<3;r++){residual[r]=gps[r]*invwidth;J[r][r]=b[21+r]*invwidth;weight[r]=1;}
    // UpVectorError applied to world up and north. The second axis is the
    // ISM extension that retains heading information for a nadir camera.
    for(uint axis=0;axis<2;axis++) {
        float3 target=axis==0?float3(0,0,1):float3(0,1,0);
        float3 predicted=R*(transpose(base)*target),r=(predicted-target)*b[24];
        float norm2=dot(r,r),w=1/(1+norm2);cost[p.u[1]+i*3+axis+1]=.5f*(norm2<1e-4f?norm2*(1-.5f*norm2):log(1+norm2));
        for(uint row=0;row<3;row++) {
            uint idx=3+axis*3+row;residual[idx]=r[row]*invwidth;weight[idx]=w;
            for(uint k=0;k<3;k++)J[idx][k+3]=cross(left[k],predicted)[row]*b[24]*invwidth;
        }
    }
    // Per-camera 8x8 Hessian and 8-vector gradient; no pixel-sized prior buffers.
    for(uint k=0;k<72;k++) {
        uint row=k<64?k/8:k-64,col=k%8;float sum=0;
        for(uint r=0;r<9;r++)sum+=J[r][row]*(k<64?J[r][col]:residual[r])*weight[r];
        priorBlocks[i*72+k]=sum;
    }
    bool valid=C.z>.02f&&C.z<100.f&&all(isfinite(C))&&all(isfinite(a));
    for(uint c=0;c<4;c++) {
        float3 ray=rays*float3((c&1)?1.f:0.f,(c&2)?b[25]:0.f,1);
        valid=valid&&ray.z<-.001f;
    }
    if(!valid)cost[p.u[1]+i*3]=INFINITY;
}
kernel void geo_visual_linearize(device const float *matrices [[buffer(0)]],device const float *derivatives [[buffer(1)]],
    device const float4 *points [[buffer(2)]],device const uint2 *owners [[buffer(3)]],
    device float *jac [[buffer(4)]],device float4 *rw [[buffer(5)]],device float *cost [[buffer(6)]],
    constant Params &p [[buffer(15)]],uint k [[thread_position_in_grid]]) {
    if(k>=p.u[0])return;bool reverse=k&1;uint2 ids=owners[k];float4 xy=points[k]/p.f[0];
    if(reverse){ids=ids.yx;xy=xy.zwxy;}
    float det;float3x3 source=geo_load(matrices+ids.x*9),target=geo_load(matrices+ids.y*9);
    float3x3 inverse=align_inverse(target,det);float3 a=float3(xy.xy,1),u=inverse*(source*a);
    float2 q=u.xy/u.z,r=q-xy.zw,sq=r*r*p.f[0]*p.f[0];
    rw[k]=float4(r,.5f*rsqrt(1+sq));cost[k]=.5f*dot(sq/(sqrt(1+sq)+1),float2(1));
    for(uint j=0;j<8;j++) {
        float3 da=inverse*(geo_load(derivatives+(ids.x*8+j)*9)*a);
        float3 db=-(inverse*(geo_load(derivatives+(ids.y*8+j)*9)*u));
        float2 ja=(da.xy-q*da.z)/u.z,jb=(db.xy-q*db.z)/u.z;
        for(uint axis=0;axis<2;axis++){
            jac[k*32+axis*16+j+(reverse?8:0)]=ja[axis];
            jac[k*32+axis*16+j+(reverse?0:8)]=jb[axis];
        }
    }
}
kernel void geo_visual_cost(device const float *matrices [[buffer(0)]],device const float4 *points [[buffer(1)]],
    device const uint2 *owners [[buffer(2)]],device float *cost [[buffer(3)]],
    constant Params &p [[buffer(15)]],uint k [[thread_position_in_grid]]) {
    if(k>=p.u[0])return;uint2 ids=owners[k];float4 xy=points[k]/p.f[0];
    if(k&1){ids=ids.yx;xy=xy.zwxy;}
    float det;float3x3 H=geo_load(matrices+ids.x*9),T=geo_load(matrices+ids.y*9);
    float3 u=align_inverse(T,det)*(H*float3(xy.xy,1));float2 r=(u.xy/u.z-xy.zw)*p.f[0],sq=r*r;
    cost[k]=abs(det)>1e-10f&&u.z>0&&all(isfinite(r))?.5f*dot(sq/(sqrt(1+sq)+1),float2(1)):INFINITY;
}
kernel void geo_assemble(device const float *blocks [[buffer(0)]],device const uint4 *edges [[buffer(1)]],
    device const int *links [[buffer(2)]],device const uint *offsets [[buffer(3)]],device const uint2 *incidence [[buffer(4)]],
    device float *normal [[buffer(5)]],device float *rhs [[buffer(6)]],device const float *state [[buffer(7)]],
    device const float *priors [[buffer(8)]],constant Params &p [[buffer(15)]],uint k [[thread_position_in_grid]]) {
    uint n=p.u[0],d=n*8;if(k>=d*d+d)return;bool grad=k>=d*d;uint row=grad?k-d*d:k/d,col=grad?row:k%d;
    uint i=row/8,j=col/8,r=row%8,c=col%8;float value=0;
    if(r>=6||c>=6)value=(!grad&&row==col)?1:0;
    else if(i==j) {
        value=priors[i*72+(grad?64+r:r*8+c)];
        for(uint a=offsets[i];a<offsets[i+1];a++){
            uint2 v=incidence[a];uint base=v.y*8;
            value+=blocks[v.x*272+(grad?256+base+r:(base+r)*16+base+c)];
        }
        if(!grad&&row==col)value+=state[0]*max(value,1e-8f)+1e-9f;
    }else{int e=links[i*n+j];if(e>=0){uint4 info=edges[e];uint rb=i==info.x?0:8,cb=j==info.x?0:8;value=blocks[e*272+(rb+r)*16+cb+c];}}
    if(grad)rhs[row]=-value;else normal[k]=value;
}

// ===== ADM fsr =====
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

// ===== ADM coherent =====
// Low-resolution observation vectors drive coherent source labels. Full-size
// output pixels always come from exactly one photograph: no color-mean blending.
kernel void coherent_observation(device const uchar *im [[buffer(0)]],device uchar4 *obs [[buffer(1)]],device half *confidence [[buffer(2)]],device float4 *mean [[buffer(3)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint w=p.u[0],h=p.u[1],gw=p.u[2],gh=p.u[3],count=gw*gh;if(i>=count)return;
    float2 xy=transform(float2(i%gw,i/gw),p);float2 d=min(xy,float2(w-1,h-1)-xy)/float2(w*.5f,h*.5f);float c=min(d.x,d.y);
    bool valid=all(isfinite(xy))&&c>=.02f;float3 rgb=0;
    if(valid){
        if(p.u[5]==4){float4 sample=fsr_bilinear(im,xy,w,h,4);valid=sample.w>.999f;rgb=rint(sample.xyz*255.f);}
        else rgb=interpolate((device const packed_uchar3*)im,xy,w,h,p.u[15]);
    }c=valid?c*c:0;
    uint j=p.u[4]*count+i;obs[j]=uchar4(uchar3(rgb),valid?255:0);confidence[j]=half(c);mean[i]+=float4(rgb*c,c);
}
kernel void coherent_normalize(device float4 *mean [[buffer(0)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i<p.u[0]){float4 m=mean[i];mean[i]=float4(m.xyz/max(m.w,1e-9f),m.w);}
}
kernel void coherent_unary(device const uchar4 *obs [[buffer(0)]],device const half *confidence [[buffer(1)]],device float4 *mean [[buffer(2)]],device half *unary [[buffer(3)]],device ushort *labels [[buffer(4)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint count=p.u[0];if(i>=count)return;float best=INFINITY;ushort id=65535;
    float variance=0;
    for(uint k=0;k<p.u[1];k++){uint j=k*count+i;float3 d=float3(obs[j].xyz)-mean[i].xyz;variance+=confidence[j]*dot(d,d);}
    // At parallax/motion disagreements, consensus can erase a real object.
    // Use geometric source confidence there, never vote for the background.
    variance/=max(3*mean[i].w,1e-9f);
    float photometric=variance<225.f?.15f:0.f;
    for(uint k=0;k<p.u[1];k++){
        uint j=k*count+i;float c=confidence[j];float3 delta=float3(obs[j].xyz)-mean[i].xyz;
        float v=obs[j].w?3*(1-sqrt(c))+photometric*min(dot(delta,delta)/(3*22*22.f),4.f)+.0001f*k:INFINITY;
        unary[j]=half(v);if(v<best){best=v;id=k;}
    }labels[i]=id;mean[i].w=variance;
}
// Fit only smooth exposure offsets. High-disagreement structures do not vote
// into the bias field, and no detail from another photograph is transferred.
kernel void coherent_bias(device const uchar4 *obs [[buffer(0)]],device const half *confidence [[buffer(1)]],device const float4 *mean [[buffer(2)]],device float4 *bias [[buffer(3)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint count=p.u[0];if(i>=count)return;uint j=p.u[1]*count+i;
    float weight=obs[j].w&&mean[i].w<900.f?float(confidence[j]):0.f;
    float3 delta=mean[i].xyz-float3(obs[j].xyz);
    bias[i]=float4(clamp(delta,-32.f,32.f)*weight,weight);
}
kernel void coherent_relax(device const half *unary [[buffer(0)]],device ushort *labels [[buffer(1)]],device const float4 *mean [[buffer(2)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint w=p.u[0],h=p.u[1],count=w*h;if(i>=count)return;uint x=i%w,y=i/w;
    if((x+y)%2!=p.u[3]||labels[i]==65535)return;
    uint neighbors[4]={x?i-1:i,x+1<w?i+1:i,y?i-w:i,y+1<h?i+w:i};float weights[4];
    for(uint q=0;q<4;q++){float3 d=mean[i].xyz-mean[neighbors[q]].xyz;weights[q]=neighbors[q]==i||labels[neighbors[q]]==65535?0:p.f[0]/(1+dot(d,d)/(3*30*30.f));}
    float best=INFINITY;ushort winner=labels[i];
    for(uint k=0;k<p.u[2];k++){
        float v=unary[k*count+i];if(!isfinite(v))continue;
        for(uint q=0;q<4;q++)v+=labels[neighbors[q]]==k?0:weights[q];
        // Keep existing label on exact ties to avoid parity-dependent flicker.
        if(v<best-1e-6f||(abs(v-best)<=1e-6f&&k==labels[i])){best=v;winner=k;}
    }labels[i]=winner;
}
kernel void coherent_render(device const uchar *im [[buffer(0)]],device const ushort *labels [[buffer(1)]],device float *best [[buffer(2)]],device uchar4 *selected [[buffer(3)]],device ushort *coverage [[buffer(4)]],device ushort *ids [[buffer(5)]],device const float4 *bias [[buffer(6)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint w=p.u[0],h=p.u[1],rw=p.u[2],rh=p.u[3];if(i>=rw*rh)return;
    uint x=i%rw+p.u[4],y=i/rw+p.u[5],j=y*p.u[6]+x;
    float2 xy=transform(float2(i%rw,i/rw),p);float2 d=min(xy,float2(w-1,h-1)-xy)/float2(w*.5f,h*.5f);float c=min(d.x,d.y);
    if(!all(isfinite(xy))||c<.02f)return;
    if(p.u[13]==4&&fsr_bilinear(im,xy,w,h,4).w<.999f)return;
    uint lx=min(uint((x+.5f)*p.u[8]/p.u[6]),p.u[8]-1),ly=min(uint((y+.5f)*p.u[9]/p.u[7]),p.u[9]-1);
    float score=c+(labels[ly*p.u[8]+lx]==p.u[10]?2.f:0.f);if(p.u[11])coverage[j]++;
    if(score>best[j]){
        float2 q=clamp((float2(x,y)+.5f)*float2(p.u[8],p.u[9])/float2(p.u[6],p.u[7])-.5f,float2(0),float2(p.u[8]-1,p.u[9]-1));
        uint2 b=uint2(q),next=min(b+1,uint2(p.u[8]-1,p.u[9]-1));float2 f=fract(q);
        float4 correction=mix(mix(bias[b.y*p.u[8]+b.x],bias[b.y*p.u[8]+next.x],f.x),mix(bias[next.y*p.u[8]+b.x],bias[next.y*p.u[8]+next.x],f.x),f.y);
        uint channels=p.u[13]==4?4:3;
        float3 rgb=p.u[12]?fsr_easu_sample(im,xy,w,h,channels).xyz*255.f:
            (channels==4?rint(fsr_bilinear(im,xy,w,h,4).xyz*255.f):interpolate((device const packed_uchar3*)im,xy,w,h,p.u[15]));
        rgb+=clamp(correction.xyz/max(correction.w,1e-6f),-32.f,32.f);
        best[j]=score;selected[j]=uchar4(uchar3(rint(clamp(rgb,0.f,255.f))),255);if(p.u[11])ids[j]=p.u[10];
    }
}
// Explicitly artistic screen-space relief, not reconstructed scene depth.
kernel void shadow_downsample(device const uchar4 *rgba [[buffer(0)]],device float4 *low [[buffer(1)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint w=p.u[0],h=p.u[1],lw=p.u[2],lh=p.u[3];if(i>=lw*lh)return;
    uint x=min(uint((i%lw+.5f)*w/lw),w-1),y=min(uint((i/lw+.5f)*h/lh),h-1);uchar4 c=rgba[y*w+x];
    float l=dot(float3(c.zyx)/255.f,float3(.2126f,.7152f,.0722f));low[i]=c.w?float4(l,l,l,1):float4(0);
}
kernel void shadow_relief(device const float4 *small [[buffer(0)]],device const float4 *large [[buffer(1)]],device float4 *relief [[buffer(2)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.u[0])return;float4 a=small[i],b=large[i];float contrast=a.x/max(a.w,1e-6f)-b.x/max(b.w,1e-6f);
    relief[i]=float4(clamp(contrast*16.f,-2.f,2.f),0,0,a.w>.5f?1:0);
}
kernel void shadow_rays(device const float4 *height [[buffer(0)]],device float4 *shade [[buffer(1)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint w=p.u[0],h=p.u[1];if(i>=w*h)return;float4 base=height[i];if(!base.w){shade[i]=float4(1);return;}
    float2 xy=float2(i%w,i/w);float occlusion=0;
    for(int ray=-1;ray<=1;ray++){
        float angle=p.f[0]+ray*.07f;float2 dir=float2(sin(angle),-cos(angle));float blocked=0;
        for(uint k=1;k<=24;k++){
            float distance=.75f*k;int2 q=int2(rint(xy+dir*distance));if(any(q<0)||q.x>=int(w)||q.y>=int(h))break;
            float4 z=height[q.y*w+q.x];if(!z.w)continue;
            blocked=max(blocked,clamp((z.x-base.x-distance*p.f[1]-.05f)*1.5f,0.f,1.f));
        }occlusion+=blocked/3;
    }shade[i]=float4(1-p.f[2]*occlusion);
}
kernel void shadow_apply(device const uchar4 *rgba [[buffer(0)]],device const float4 *shade [[buffer(1)]],device uchar4 *output [[buffer(2)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint w=p.u[0],h=p.u[1],sw=p.u[2],sh=p.u[3];if(i>=w*h)return;
    float2 q=clamp((float2(i%w,i/w)+.5f)*float2(sw,sh)/float2(w,h)-.5f,float2(0),float2(sw-1,sh-1));uint2 b=uint2(q),n=min(b+1,uint2(sw-1,sh-1));float2 f=fract(q);
    float factor=mix(mix(shade[b.y*sw+b.x].x,shade[b.y*sw+n.x].x,f.x),mix(shade[n.y*sw+b.x].x,shade[n.y*sw+n.x].x,f.x),f.y);
    uchar4 c=rgba[i];float3 rgb=decode(float3(c.xyz)/255.f)*factor;output[i]=uchar4(uchar3(rint(clamp(encode(rgb),0.f,1.f)*255)),c.w);
}
kernel void flat_light_encoded(device const uchar4 *rgba [[buffer(0)]],device uchar4 *encoded [[buffer(1)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.u[0])return;uchar4 c=rgba[i];float3 rgb=decode(float3(c.zyx)/255.f);
    float3 v=c.w?shade(rgb,float3(0,0,1),1,float3(p.f[0],p.f[1],p.f[2]),p.f[3]):float3(0);
    encoded[i]=uchar4(uchar3(rint(clamp(encode(v.zyx),0.f,1.f)*255.f)),c.w);
}

// ===== ADM lasso =====
// Conservative polygon crop. All mask analysis and pixel decisions run on GPU.
// Coarse mask bits: 1 = entirely opaque, 2 = contains no near-black pixels.
kernel void lasso_mask(device const uchar4 *image [[buffer(0)]],
                       device uchar *mask [[buffer(1)]],
                       constant Params &p [[buffer(15)]],
                       uint i [[thread_position_in_grid]]) {
    uint w=p.u[0], h=p.u[1], stride=p.u[2], gw=p.u[3], gh=p.u[4];
    if(i>=gw*gh) return;
    uint x0=(i%gw)*stride, y0=(i/gw)*stride;
    bool opaque=true, nonblack=true;
    for(uint y=y0;y<min(y0+stride,h);y++)
        for(uint x=x0;x<min(x0+stride,w);x++) {
            uchar4 c=image[y*w+x];
            opaque=opaque&&c.w==255;
            nonblack=nonblack&&max(max(c.x,c.y),c.z)>8;
        }
    mask[i]=(opaque?1:0)|(nonblack?2:0);
}

kernel void lasso_morph(device const uchar *input [[buffer(0)]],
                        device uchar *output [[buffer(1)]],
                        constant Params &p [[buffer(15)]],
                        uint i [[thread_position_in_grid]]) {
    if(i>=p.u[0]*p.u[1]) return;
    int x=int(i%p.u[0]), y=int(i/p.u[0]), radius=int(p.u[4]);
    bool value=p.u[3]==0;
    for(int offset=-radius;offset<=radius;offset++) {
        int xx=x+(p.u[2]==0?offset:0), yy=y+(p.u[2]?offset:0);
        bool sample=xx>=0&&yy>=0&&xx<int(p.u[0])&&yy<int(p.u[1])&&(input[yy*p.u[0]+xx]&1);
        value=p.u[3]?(value||sample):(value&&sample);
    }
    output[i]=(value?1:0)|(input[i]&2);
}

kernel void lasso_rows(device const uchar *mask [[buffer(0)]],
                       device float4 *rows [[buffer(1)]],
                       constant Params &p [[buffer(15)]],
                       uint y [[thread_position_in_grid]]) {
    uint w=p.u[0]; if(y>=p.u[1]) return;
    float sx=0, count=0;
    for(uint x=0;x<w;x++) if(mask[y*w+x]&1) {sx+=x+.5f; count++;}
    rows[y]=float4(sx,(y+.5f)*count,count,0);
}

kernel void lasso_center(device const float4 *rows [[buffer(0)]],
                         device const uchar *mask [[buffer(1)]],
                         device float4 *center [[buffer(2)]],
                         constant Params &p [[buffer(15)]],
                         uint i [[thread_position_in_grid]]) {
    if(i) return;
    float4 sum=0;
    for(uint y=0;y<p.u[1];y++) sum+=rows[y];
    float2 c=sum.xy/max(sum.z,1.f);
    uint x=min(uint(c.x),p.u[0]-1), y=min(uint(c.y),p.u[1]-1);
    center[0]=float4(c,sum.z,sum.z>0&&(mask[y*p.u[0]+x]&1)?1:0);
}

inline float2 lasso_direction(uint i,uint count) {
    float angle=6.28318530718f*i/count;
    return float2(cos(angle),sin(angle));
}

inline uchar lasso_sample(device const uchar *mask,float2 q,uint w,uint h) {
    if(any(q<0)||q.x>=w||q.y>=h) return 0;
    return mask[uint(q.y)*w+uint(q.x)];
}

kernel void lasso_rays(device const uchar *mask [[buffer(0)]],
                       device const float4 *center [[buffer(1)]],
                       device float *radii [[buffer(2)]],
                       constant Params &p [[buffer(15)]],
                       uint i [[thread_position_in_grid]]) {
    if(i>=p.u[2]) return;
    float2 c=center[0].xy, direction=lasso_direction(i,p.u[2]);
    float radius=0, limit=length(float2(p.u[0],p.u[1]));
    if(center[0].w) {
        for(float r=.5f;r<limit;r+=.5f) {
            if(!(lasso_sample(mask,c+direction*r,p.u[0],p.u[1])&1)) break;
            radius=r;
        }
        // Only inspect black strips close to the outer perimeter. Interior
        // dark roofs/cars/shadows do not become holes or determine the lasso.
        for(float r=max(.5f,radius-3*p.f[0]);r<=radius;r+=.5f)
            if(!(lasso_sample(mask,c+direction*r,p.u[0],p.u[1])&2)) {
                radius=r-.5f; break;
            }
    }
    radii[i]=max(0.f,radius-p.f[0]);
}

// Circular morphological opening removes isolated outward plateaus without
// flattening the long monotonic portions of the footprint boundary.
kernel void lasso_open(device const float *input [[buffer(0)]],
                       device float *output [[buffer(1)]],
                       constant Params &p [[buffer(15)]],
                       uint i [[thread_position_in_grid]]) {
    if(i>=p.u[0]) return;
    float value=p.u[1]?0.f:INFINITY;
    for(int j=-int(p.u[2]);j<=int(p.u[2]);j++) {
        float r=input[(int(i)+j+int(p.u[0]))%int(p.u[0])];
        value=p.u[1]?max(value,r):min(value,r);
    }
    output[i]=value;
}

kernel void lasso_anchors(device const float *rays [[buffer(0)]],
                          device float *anchors [[buffer(1)]],
                          constant Params &p [[buffer(15)]],
                          uint i [[thread_position_in_grid]]) {
    if(i>=p.u[1]) return;
    int n=int(p.u[0]), center=int(i*p.u[0]/p.u[1]);
    float r=INFINITY;
    for(int j=-3;j<=3;j++) r=min(r,rays[(center+j+n)%n]);
    anchors[i]=r;
}

inline bool lasso_segment_valid(device const uchar *mask,float2 a,float2 b,uint w,uint h) {
    uint count=max(1u,uint(ceil(length(b-a)*2)));
    for(uint j=0;j<=count;j++)
        if(!(lasso_sample(mask,mix(a,b,float(j)/count),w,h)&1)) return false;
    return true;
}

kernel void lasso_refine(device const uchar *mask [[buffer(0)]],
                         device const float4 *center [[buffer(1)]],
                         device const float *input [[buffer(2)]],
                         device float *output [[buffer(3)]],
                         constant Params &p [[buffer(15)]],
                         uint i [[thread_position_in_grid]]) {
    uint n=p.u[2]; if(i>=n) return;
    uint left=(i+n-1)%n, right=(i+1)%n;
    float radius=min(input[i],(input[left]+2*input[i]+input[right])*.25f);
    float2 c=center[0].xy, a=c+lasso_direction(left,n)*input[left];
    float2 b=c+lasso_direction(i,n)*radius, d=c+lasso_direction(right,n)*input[right];
    bool valid=lasso_segment_valid(mask,a,b,p.u[0],p.u[1])&&lasso_segment_valid(mask,b,d,p.u[0],p.u[1]);
    output[i]=valid?radius:radius*.98f;
}

kernel void lasso_apply(device uchar4 *image [[buffer(0)]],
                        device const float4 *center [[buffer(1)]],
                        device const float *radii [[buffer(2)]],
                        device ushort *coverage [[buffer(3)]],
                        device ushort *source_ids [[buffer(4)]],
                        device atomic_uint *counts [[buffer(5)]],
                        constant Params &p [[buffer(15)]],
                        uint i [[thread_position_in_grid]],
                        uint t [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
    uint w=p.u[0], h=p.u[1];
    uint was_valid=0, kept=0;
    if(i<w*h) {
        was_valid=image[i].w>0;
        float2 q=(float2(i%w,i/w)+.5f)/p.u[2]-center[0].xy;
        float angle=atan2(q.y,q.x); if(angle<0) angle+=6.28318530718f;
        uint a=min(uint(angle*p.u[3]/6.28318530718f),p.u[3]-1), b=(a+1)%p.u[3];
        float2 va=lasso_direction(a,p.u[3])*radii[a], vb=lasso_direction(b,p.u[3])*radii[b];
        float2 edge=vb-va, delta=q-va;
        bool inside=edge.x*delta.y-edge.y*delta.x>=0;
        if(center[0].w&&!inside) {
            image[i]=uchar4(0);
            if(p.u[4]) coverage[i]=0;
            if(p.u[5]) source_ids[i]=65535;
        }
        kept=image[i].w>0;
    }
    uint2 sums=simd_sum(uint2(was_valid,kept));
    threadgroup uint2 partial[8];
    if(lane==0) partial[sg]=sums;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t==0) {
        uint2 sum=0; for(uint j=0;j<8;j++) sum+=partial[j];
        atomic_fetch_add_explicit(counts,sum.x,memory_order_relaxed);
        atomic_fetch_add_explicit(counts+1,sum.y,memory_order_relaxed);
    }
}

// ===== ADM presentation =====
// Metal ports of the supplied presentation recipe. OpenCV/Pillow numerical
// conventions follow the pinned references; see third_party/PRESENTATION_NOTICES.txt.
inline uchar3 post_byte(float3 v){return uchar3(clamp(rint(v),0.f,255.f));}
inline uchar4 post_at(device const uchar4 *a,int x,int y,uint w,uint h){return a[reflect101(y,h)*w+reflect101(x,w)];}
kernel void post_lab(device const uchar4 *a [[buffer(0)]],device uchar4 *b [[buffer(1)]],device const int *tab [[buffer(2)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]){
    if(i>=p.u[0])return;uchar4 c=a[i];int3 v;
    if(!p.u[1]){
        int3 rgb=int3(tab[c.z],tab[c.y],tab[c.x]);int f[3];
        for(int j=0;j<3;j++){int k=256+3072+j*3;int q=(rgb.x*tab[k]+rgb.y*tab[k+1]+rgb.z*tab[k+2]+2048)>>12;f[j]=tab[256+q];}
        v=int3((296*f[1]-((16*255*32768+50)/100)+16384)>>15,(500*(f[0]-f[1])+128*32768+16384)>>15,(200*(f[1]-f[2])+128*32768+16384)>>15);
    }else{
        int y=tab[c.x*2],fy=tab[c.x*2+1];
        int ad=((5*int(c.y)*53687+128)>>13)-128*16384/500;
        int bd=((int(c.z)*41943+16)>>9)-128*16384/200+1;
        int x=tab[512+fy+ad+8145],z=tab[512+fy-bd+8145];
        for(int j=0;j<3;j++){int k=512+36864+4096+j*3;int q=clamp((tab[k]*x+tab[k+1]*y+tab[k+2]*z+8192)>>14,0,4095);v[2-j]=tab[512+36864+q];}
    }b[i]=uchar4(uchar3(clamp(v,0,255)),c.w);
}
// Same quantized patch-distance and integer weights as OpenCV's 7x7 / 15x15
// NLM. Early rejection only skips weights that are exactly zero in its table.
kernel void post_nlm(device const uchar4 *a [[buffer(0)]],device uchar4 *b [[buffer(1)]],device const uint *lut [[buffer(2)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]){
    uint w=p.u[0],h=p.u[1];if(i>=w*h)return;int x=i%w,y=i/w;
    uint3 estimate=0;uint2 weight=0;
    for(int dy=-7;dy<=7;dy++)for(int dx=-7;dx<=7;dx++){
        uint2 dist=0;
        if(dx||dy){
            bool reject=false;
            for(int py=-3;py<=3&&!reject;py++)for(int px=-3;px<=3;px++){
                int3 d=int3(post_at(a,x+px,y+py,w,h).xyz)-int3(post_at(a,x+dx+px,y+dy+py,w,h).xyz);
                dist+=uint2(d.x*d.x,d.y*d.y+d.z*d.z);
                if(dist.x>=p.u[2]&&dist.y>=p.u[3]){reject=true;break;}
            }
        }
        uint2 q=dist>>6;uint2 v=uint2(q.x<64?lut[q.x]:0,q.y<64?lut[64+q.y]:0);
        uint3 c=uint3(post_at(a,x+dx,y+dy,w,h).xyz);
        estimate+=c*uint3(v.x,v.y,v.y);weight+=v;
    }
    uint3 divisor=uint3(weight.x,weight.y,weight.y);b[i]=uchar4(uchar3((estimate+divisor/2)/divisor),a[i].w);
}
inline int post_reflect(int x,int n){return n>10?n-1-abs(n-1-abs(x)):reflect101(x,n);}
kernel void post_nlm_rows(device const uchar4 *a [[buffer(0)]],device ushort2 *rows [[buffer(1)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]){
    uint w=p.u[0],h=p.u[1];if(i>=w*(h+6))return;int x=i%w,y=int(i/w)-3,dx=int(p.u[2]),dy=int(p.u[3]);
    uint ay=post_reflect(y,h)*w,by=post_reflect(y+dy,h)*w;uint2 sum=0;
    for(int k=-3;k<=3;k++){
        int3 d=int3(a[ay+post_reflect(x+k,w)].xyz)-int3(a[by+post_reflect(x+k+dx,w)].xyz);
        sum+=uint2(d.x*d.x,d.y*d.y+d.z*d.z);
    }
    rows[i]=ushort2(min(sum,uint2(p.u[4],p.u[5])));
}
kernel void post_nlm_accumulate(device const uchar4 *a [[buffer(0)]],device const ushort2 *rows [[buffer(1)]],device uint4 *sum [[buffer(2)]],device uint *chroma_weight [[buffer(3)]],device const uint *lut [[buffer(4)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]){
    uint w=p.u[0],h=p.u[1];if(i>=w*h)return;uint2 dist=0;
    for(uint k=0;k<7;k++)dist+=uint2(rows[i+k*w]);
    uint2 q=dist>>6,v=uint2(q.x<64?lut[q.x]:0,q.y<64?lut[64+q.y]:0);if(!any(v!=0))return;
    int x=int(i%w)+int(p.u[2]),y=int(i/w)+int(p.u[3]);uint3 c=uint3(a[post_reflect(y,h)*w+post_reflect(x,w)].xyz);
    sum[i]+=uint4(c*uint3(v.x,v.y,v.y),v.x);chroma_weight[i]+=v.y;
}
kernel void post_nlm_finish(device const uchar4 *a [[buffer(0)]],device const uint4 *sum [[buffer(1)]],device const uint *chroma_weight [[buffer(2)]],device uchar4 *b [[buffer(3)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]){
    if(i>=p.u[0])return;uint4 s=sum[i];uint3 d=uint3(s.w,chroma_weight[i],chroma_weight[i]);b[i]=uchar4(uchar3((s.xyz+d/2)/d),a[i].w);
}
kernel void post_lanczos_x(device const uchar4 *a [[buffer(0)]],device int4 *b [[buffer(1)]],device const int *idx [[buffer(2)]],device const int *coeff [[buffer(3)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]){
    uint w=p.u[0],h=p.u[1],ow=p.u[2];if(i>=ow*h)return;uint x=i%ow,y=i/ow;int4 v=0;
    for(uint k=0;k<8;k++)v+=int4(a[y*w+idx[x*8+k]])*coeff[x*8+k];b[i]=v;
}
kernel void post_lanczos_y(device const int4 *a [[buffer(0)]],device uchar4 *b [[buffer(1)]],device const int *idx [[buffer(2)]],device const int *coeff [[buffer(3)]],device const uchar4 *original [[buffer(4)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]){
    uint ow=p.u[2],oh=p.u[3];if(i>=ow*oh)return;uint x=i%ow,y=i/ow;int4 v=0;
    for(uint k=0;k<8;k++)v+=a[idx[y*8+k]*ow+x]*coeff[y*8+k];
    uchar4 c=uchar4(clamp((v+(1<<21))>>22,0,255));c.w=original[min(y*p.u[1]/oh,p.u[1]-1)*p.u[0]+min(x*p.u[0]/ow,p.u[0]-1)].w;b[i]=c;
}
kernel void post_gauss_x(device const uchar4 *a [[buffer(0)]],device ushort4 *b [[buffer(1)]],device const uint *weight [[buffer(2)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]){
    uint w=p.u[0],h=p.u[1];if(i>=w*h)return;uint4 v=0;int r=p.u[2];
    for(int k=-r;k<=r;k++)v+=uint4(post_at(a,int(i%w)+k,i/w,w,h))*weight[k+r];b[i]=ushort4(v);
}
kernel void post_gauss_y(device const ushort4 *a [[buffer(0)]],device uchar4 *b [[buffer(1)]],device const uint *weight [[buffer(2)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]){
    uint w=p.u[0],h=p.u[1];if(i>=w*h)return;uint4 v=0;int r=p.u[2];
    for(int k=-r;k<=r;k++)v+=uint4(a[reflect101(int(i/w)+k,h)*w+i%w])*weight[k+r];b[i]=uchar4(min((v+32768)>>16,255u));
}
kernel void post_detail(device const uchar4 *a [[buffer(0)]],device const uchar4 *blur [[buffer(1)]],device uchar4 *b [[buffer(2)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]){
    if(i>=p.u[0])return;float3 x=float3(a[i].xyz),low=float3(blur[i].xyz);
    b[i]=uchar4(post_byte(p.u[1]?(1+p.f[0])*x-p.f[0]*low:x+p.f[0]*max(x-low,0.f)),a[i].w);
}
kernel void post_contrast(device uchar4 *a [[buffer(0)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]){
    if(i>=p.u[0])return;uchar4 c=a[i];c.xyz=c.w?post_byte(abs(float3(c.xyz)*p.f[0]+p.f[1])):uchar3(0);a[i]=c;
}
kernel void post_clahe_blend(device const uchar4 *source [[buffer(0)]],device const uchar4 *graded [[buffer(1)]],device uchar4 *output [[buffer(2)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]){
    if(i>=p.u[0])return;output[i]=uchar4(post_byte(float3(source[i].xyz)*(1-p.f[0])+float3(graded[i].xyz)*p.f[0]),source[i].w);
}
kernel void post_chroma_clean(device const uchar4 *source [[buffer(0)]],device uchar4 *output [[buffer(1)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]){
    uint w=p.u[0],h=p.u[1];if(i>=w*h)return;uchar4 c=source[i];if(!c.w){output[i]=uchar4(0);return;}
    const float3 Y=float3(.299f,.587f,.114f);float3 rgb=float3(c.zyx)/255.f;float l=dot(rgb,Y);float3 sum=0;float total=0;
    for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
        uchar4 n=post_at(source,int(i%w)+dx,int(i/w)+dy,w,h);if(!n.w)continue;
        float3 color=float3(n.zyx)/255.f;float nl=dot(color,Y),d=(nl-l)/.08f;
        float weight=exp(-d*d)*exp(-float(dx*dx+dy*dy)*.5f);sum+=(color-nl)*weight;total+=weight;
    }
    float3 chroma=(rgb-l)*(1-p.f[0])+sum/max(total,1e-6f)*p.f[0];
    float pos=max(max(chroma.x,max(chroma.y,chroma.z)),1e-6f),neg=max(-min(chroma.x,min(chroma.y,chroma.z)),1e-6f);
    rgb=l+chroma*min(1.f,min((1-l)/pos,l/neg));output[i]=uchar4(uchar3(clamp(rgb.zyx,0.f,1.f)*255.f+.5f),c.w);
}
// Parallel histogram, one workgroup per CLAHE tile; all arithmetic is integer
// through clipping and redistribution, matching OpenCV's 8-bit LUT behavior.
kernel void post_clahe_lut(device const uchar4 *a [[buffer(0)]],device uchar *lut [[buffer(1)]],constant Params &p [[buffer(15)]],uint g [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]]){
    if(g>=64)return;threadgroup atomic_uint histogram[256];threadgroup uint clipped[256];
    atomic_store_explicit(&histogram[t],0,memory_order_relaxed);threadgroup_barrier(mem_flags::mem_threadgroup);
    uint w=p.u[0],h=p.u[1],tw=p.u[2],th=p.u[3],area=tw*th,limit=max(area*2/256,1u);
    for(uint j=t;j<area;j+=256){uint x=reflect101(int((g%8)*tw+j%tw),w),y=reflect101(int((g/8)*th+j/tw),h);atomic_fetch_add_explicit(&histogram[a[y*w+x].x],1,memory_order_relaxed);}
    threadgroup_barrier(mem_flags::mem_threadgroup);uint n=atomic_load_explicit(&histogram[t],memory_order_relaxed);clipped[t]=n>limit?n-limit:0;
    atomic_store_explicit(&histogram[t],min(n,limit),memory_order_relaxed);threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t==0){uint total=0;for(uint k=0;k<256;k++)total+=clipped[k];uint batch=total/256,residual=total%256,step=max(256/max(residual,1u),1u),sum=0;
        for(uint k=0;k<256;k++){sum+=atomic_load_explicit(&histogram[k],memory_order_relaxed)+batch+uint(residual&&k%step==0&&k/step<residual);lut[g*256+k]=uchar(clamp(rint(float(sum)*(255.f/area)),0.f,255.f));}}
}
kernel void post_clahe_apply(device const uchar4 *a [[buffer(0)]],device uchar4 *b [[buffer(1)]],device const uchar *lut [[buffer(2)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]){
    uint w=p.u[0],h=p.u[1];if(i>=w*h)return;float2 q=float2(i%w,i/w)/float2(p.u[2],p.u[3])-.5f;int2 base=int2(floor(q));float2 f=q-floor(q);int2 lo=max(base,0),hi=min(base+1,7);uchar4 c=a[i];
    c.x=uchar(clamp(rint(mix(mix(float(lut[(lo.y*8+lo.x)*256+c.x]),float(lut[(lo.y*8+hi.x)*256+c.x]),f.x),mix(float(lut[(hi.y*8+lo.x)*256+c.x]),float(lut[(hi.y*8+hi.x)*256+c.x]),f.x),f.y)),0.f,255.f));b[i]=c;
}
// PIL-compatible byte HSV conventions. See Pillow 12 libImaging/Convert.c;
// the supplied recipe deliberately quantizes when classifying/rotating hue.
inline float3 post_rgb_hsv(uchar3 rgb) {
    float3 x=float3(rgb);float mx=max(x.x,max(x.y,x.z)),mn=min(x.x,min(x.y,x.z));
    if(mx==mn)return float3(0,0,mx);
    float d=mx-mn;float3 rc=(mx-x)/d;
    float h=x.x==mx?rc.z-rc.y:(x.y==mx?2.f+rc.x-rc.z:4.f+rc.y-rc.x);
    h=fract(h/6.f+1.f);
    return float3(uchar(h*255.f),uchar((d/mx)*255.f),mx);
}
inline float3 post_hsv_rgb(float3 hsv) {
    float sector=hsv.x*6.f/255.f;int j=int(floor(sector));float f=sector-j,s=hsv.y/255.f,v=hsv.z;
    float low=round(v*(1.f-s)),q=round(v*(1.f-s*f)),t=round(v*(1.f-s*(1.f-f)));
    switch(j%6){case 0:return float3(v,t,low);case 1:return float3(q,v,low);
        case 2:return float3(low,v,t);case 3:return float3(low,q,v);
        case 4:return float3(t,low,v);default:return float3(v,low,q);}
}
inline float post_luma(float3 x){return x.x*.299f+x.y*.587f+x.z*.114f;}
inline float3 post_recipe(float3 x,constant float *v) {
    float3 lift=float3(v[0],v[1],v[2]),gamma=float3(v[3],v[4],v[5]),gain=float3(v[6],v[7],v[8]);
    x=pow(clamp(x*gain+lift,1e-6f,1.f),1.f/gamma);x+=float3(v[17]*.5f,0,-v[17]*.5f);
    if(v[19]!=0.f){float3 hsv=post_rgb_hsv(uchar3(clamp(x,0.f,1.f)*255.f+.5f));
        hsv.x=floor(hsv.x+v[19]/360.f*255.f-floor((hsv.x+v[19]/360.f*255.f)/255.f)*255.f);
        x=post_hsv_rgb(hsv)/255.f;}
    float l=clamp(post_luma(x),0.f,1.f);
    x+=float3(v[9],v[10],v[11])*pow(1-l,1.5f)+float3(v[12],v[13],v[14])*pow(l,1.5f);
    l=post_luma(x);float a=(l-.25f)/.25f,b=(l-.75f)/.25f;
    x+=v[22]*clamp(1-l/.3f,0.f,1.f)+v[23]*exp(-a*a)+v[24]*exp(-b*b)+v[25]*clamp((l-.7f)/.3f,0.f,1.f);
    if(v[15]>0)x+=v[15]*(x-.5f)*(1-abs(2*x-1));
    l=post_luma(x);return l+(x-l)*v[16];
}
inline uchar4 post_grade_finish(float3 x,uchar alpha,uint i,constant Params &p,device const float *noise) {
    float2 size=float2(p.u[0],p.u[1]),q=(float2(i%p.u[0],i/p.u[0])-size*.5f)/(size*.5f);
    if(p.f[18]>0)x*=clamp(1-p.f[18]*pow(max(length(q)-.3f,0.f),1.8f),0.f,1.f);
    if(p.u[2])x+=noise[i];
    return uchar4(uchar3(clamp(x.zyx,0.f,1.f)*255.f+.5f),alpha);
}
kernel void post_grade(device const uchar4 *a [[buffer(0)]],device uchar4 *b [[buffer(1)]],
                       device const float *noise [[buffer(2)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]){
    if(i>=p.u[0]*p.u[1])return;float3 x=float3(a[i].zyx)/255.f*exp2(p.f[21]);
    b[i]=post_grade_finish(post_recipe(x,p.f),a[i].w,i,p,noise);
}
// Four category scores share one uchar4 buffer and six vectorized box passes.
kernel void post_region_scores(device const uchar4 *a [[buffer(0)]],device uchar4 *scores [[buffer(1)]],
                               constant float *detect [[buffer(2)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]){
    if(i>=p.u[0]*p.u[1])return;
    float3 x=float3(a[i].zyx)/255.f*exp2(p.f[21]);float3 hsv=post_rgb_hsv(uchar3(clamp(x,0.f,1.f)*255.f));
    uchar4 result;
    for(uint k=0;k<4;k++){
        constant float *d=detect+6*k;float score;
        if(d[0]<0)score=1-smoothstep(d[2],d[2]+d[3],hsv.y);
        else{float delta=abs(hsv.x-d[0]);delta=min(delta,256-delta)/d[1];
            score=exp(-delta*delta)*smoothstep(d[2]-d[3],d[2],hsv.y);}
        score*=smoothstep(d[4]-d[5],d[4],hsv.z);result[k]=uchar(clamp(score,0.f,1.f)*255.f);
    }
    scores[i]=result;
}
kernel void post_grade_regional(device const uchar4 *a [[buffer(0)]],device uchar4 *b [[buffer(1)]],
                                device const float *noise [[buffer(2)]],device const uchar4 *scores [[buffer(3)]],
                                constant float *recipes [[buffer(4)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]){
    if(i>=p.u[0]*p.u[1])return;float3 x=float3(a[i].zyx)/255.f*exp2(p.f[21]);
    float4 weights=float4(scores[i])/255.f;float total=weights.x+weights.y+weights.z+weights.w+.12f;
    float3 out=0;for(uint k=0;k<4;k++)out+=post_recipe(x,recipes+32*k)*(weights[k]/total);
    out+=post_recipe(x,recipes+128)*(.12f/total);
    b[i]=post_grade_finish(out,a[i].w,i,p,noise);
}
// Pillow's three fractional box passes use 24-bit integer weights. Chunked
// sliding sums preserve those exact results while making each chunk independent.
kernel void post_box(device const uchar4 *a [[buffer(0)]],device uchar4 *b [[buffer(1)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]){
    uint w=p.u[0],h=p.u[1],axis=p.u[2],r=p.u[3],ww=p.u[4],fw=p.u[5];uint n=axis?h:w,lines=axis?w:h,chunks=(n+127)/128;
    if(i>=lines*chunks)return;uint line=i/chunks,start=(i%chunks)*128,end=min(start+128,n),stride=axis?w:1,base=axis?line:line*w;
    uint4 acc=0;for(int k=-int(r);k<=int(r);k++)acc+=uint4(a[base+clamp(int(start)+k,0,int(n)-1)*stride]);
    for(uint x=start;x<end;x++){
        uint left=base+clamp(int(x)-int(r)-1,0,int(n)-1)*stride,right=base+min(x+r+1,n-1)*stride;
        uint4 bulk=acc*ww+(uint4(a[left])+uint4(a[right]))*fw;b[base+x*stride]=uchar4((bulk+(1u<<23))>>24);
        acc-=uint4(a[base+max(int(x)-int(r),0)*stride]);acc+=uint4(a[right]);
    }
}
kernel void post_unsharp(device const uchar4 *a [[buffer(0)]],device const uchar4 *blur [[buffer(1)]],device uchar4 *b [[buffer(2)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]){
    if(i>=p.u[0])return;int3 x=int3(a[i].xyz),d=x-int3(blur[i].xyz);b[i]=uchar4(uchar3(clamp(x+select(int3(0),d*int(p.u[1])/100,abs(d)>2),0,255)),a[i].w);
}
kernel void post_grade_alpha(device const uchar4 *before [[buffer(0)]],device const uchar4 *blur [[buffer(1)]],device uchar4 *after [[buffer(2)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]){
    if(i>=p.u[0])return;uchar4 c=after[i];if(blur[i].w<255)c=before[i];if(!c.w)c.xyz=0;after[i]=c;
}
// Monotone highlight compression: shadows below .5 stay unchanged, neutral
// white reaches 1-reduction. Vibrance attenuates muted chroma adaptively.
kernel void post_finish_tone(device const uchar4 *source [[buffer(0)]],device uchar4 *output [[buffer(1)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]){
    if(i>=p.u[0])return;uchar4 c=source[i];float3 rgb=float3(c.zyx)/255.f;
    float l=dot(rgb,float3(.299f,.587f,.114f));
    if(l>.5f){float t=(l-.5f)*2,k=p.f[0]/(.5f-p.f[0]);float target=.5f+.5f*t/(1+k*t);rgb*=target/l;}
    float mx=max(rgb.x,max(rgb.y,rgb.z)),mn=min(rgb.x,min(rgb.y,rgb.z));
    float saturation=(mx-mn)/max(mx,1e-6f);l=dot(rgb,float3(.299f,.587f,.114f));
    rgb=l+(rgb-l)*(1-p.f[1]*(1-saturation));
    output[i]=uchar4(c.w?uchar3(clamp(rgb.zyx,0.f,1.f)*255.f+.5f):uchar3(0),c.w);
}

// ===== ADM raytracing =====
using namespace raytracing;

// Bounded two-bounce diffuse path tracer. Camera, visibility and bounce rays
// intersect a real triangle acceleration structure. No intensity-derived depth.
inline float3 rt_normal(device const float4 *v,uint primitive,float3 direction){
    uint k=primitive*3;float3 n=normalize(cross(v[k+1].xyz-v[k].xyz,v[k+2].xyz-v[k].xyz));
    return dot(n,direction)>0?-n:n;
}
inline float3 rt_basis(float3 n,float3 local){
    float3 a=normalize(cross(abs(n.z)<.99f?float3(0,0,1):float3(0,1,0),n));
    return a*local.x+cross(n,a)*local.y+n*local.z;
}
kernel void cinematic_path_trace(device const float4 *vertices [[buffer(0)]],
    device const float4 *materials [[buffer(1)]],device float4 *output [[buffer(2)]],
    primitive_acceleration_structure scene [[buffer(3)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]){
    uint w=p.u[0],h=p.u[1],samples=p.u[2],bounces=p.u[3];if(i>=w*h)return;
    float3 sun=normalize(float3(p.f[0],p.f[1],p.f[2]));
    const float3 skyColor=float3(.97f,1.f,1.04f),sunColor=float3(1.08f,1.f,.87f);
    float3 reference=.75f*skyColor+.25f*sunColor;
    intersector<triangle_data> query;query.assume_geometry_type(geometry_type::triangle);
    query.force_opacity(forced_opacity::opaque);
    ray camera;camera.origin=float3(((float(i%w)+.5f)/w-.5f)*p.f[3],(.5f-(float(i/w)+.5f)/h)*p.f[4],p.f[5]);
    camera.direction=float3(0,0,-1);camera.min_distance=.0001f;camera.max_distance=INFINITY;
    auto primary=query.intersect(camera,scene);
    if(primary.type==intersection_type::none){output[i]=float4(1,1,1,0);return;}
    float3 position=camera.origin+camera.direction*primary.distance;
    float3 normal=rt_normal(vertices,primary.primitive_id,camera.direction),sum=0;
    for(uint s=0;s<samples;s++){
        float3 origin=position,n=normal,throughput=1,radiance=0;
        for(uint bounce=0;bounce<bounces;bounce++){
            // Deterministic low-discrepancy quadrature; no temporal noise/cache.
            float u=(float(s)+.5f)/samples,v=fract((float(s)+.5f)*.61803398875f+bounce*.38196601125f);
            float phi=6.28318530718f*v;
            float disk=.00465f*sqrt(u);float3 light=normalize(sun+rt_basis(sun,float3(cos(phi)*disk,sin(phi)*disk,0)));
            ray shadow;shadow.origin=origin+n*.0001f;shadow.direction=light;shadow.min_distance=.0001f;shadow.max_distance=INFINITY;
            float cosine=max(dot(n,light),0.f);
            if(cosine>0&&query.intersect(shadow,scene).type==intersection_type::none)
                radiance+=throughput*.25f*sunColor*cosine/max(sun.z,.001f);
            ray next;next.origin=shadow.origin;next.direction=rt_basis(n,float3(sqrt(u)*cos(phi),sqrt(u)*sin(phi),sqrt(1-u)));
            next.min_distance=.0001f;next.max_distance=INFINITY;
            auto hit=query.intersect(next,scene);
            if(hit.type==intersection_type::none){radiance+=throughput*.75f*skyColor;break;}
            origin=next.origin+next.direction*hit.distance;
            n=rt_normal(vertices,hit.primitive_id,next.direction);
            throughput*=materials[hit.primitive_id].xyz;
        }
        sum+=radiance;
    }
    output[i]=float4(sum/float(samples)/reference,1);
}

kernel void post_ray_light(device const uchar4 *input [[buffer(0)]],device const float4 *light [[buffer(1)]],
    device uchar4 *output [[buffer(2)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]){
    uint w=p.u[0],h=p.u[1],lw=p.u[2],lh=p.u[3];if(i>=w*h)return;uchar4 c=input[i];
    if(!c.w){output[i]=uchar4(0);return;}
    float2 q=(float2(i%w,i/w)+.5f)*float2(lw,lh)/float2(w,h)-.5f;int2 lo=int2(floor(q));float2 f=q-floor(q);
    uint2 a=uint2(clamp(lo,int2(0),int2(lw-1,lh-1))),b=uint2(clamp(lo+1,int2(0),int2(lw-1,lh-1)));
    float3 factor=mix(mix(light[a.y*lw+a.x].xyz,light[a.y*lw+b.x].xyz,f.x),mix(light[b.y*lw+a.x].xyz,light[b.y*lw+b.x].xyz,f.x),f.y);
    factor=mix(float3(1),clamp(factor,.65f,1.25f),p.f[0]);float3 rgb=float3(c.zyx)/255.f;
    float3 linear=select(pow((rgb+.055f)/1.055f,float3(2.4f)),rgb/12.92f,rgb<=.04045f)*factor;
    rgb=select(1.055f*pow(max(linear,0.f),float3(1.f/2.4f))-.055f,12.92f*linear,linear<=.0031308f);
    output[i]=uchar4(uchar3(clamp(rgb.zyx,0.f,1.f)*255.f+.5f),c.w);
}

/*
This file is part of the FidelityFX SDK.

Copyright (C) 2024 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

// ===== AMD FSR compute_game_vector_field_inpainting_pyramid =====
namespace ism_compute_game_vector_field_inpainting_pyramid {
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wmissing-braces"


using namespace metal;

template<typename T, size_t Num>
struct spvUnsafeArray
{
    T elements[Num ? Num : 1];
    
    thread T& operator [] (size_t pos) thread
    {
        return elements[pos];
    }
    constexpr const thread T& operator [] (size_t pos) const thread
    {
        return elements[pos];
    }
    
    device T& operator [] (size_t pos) device
    {
        return elements[pos];
    }
    constexpr const device T& operator [] (size_t pos) const device
    {
        return elements[pos];
    }
    
    constexpr const constant T& operator [] (size_t pos) const constant
    {
        return elements[pos];
    }
    
    threadgroup T& operator [] (size_t pos) threadgroup
    {
        return elements[pos];
    }
    constexpr const threadgroup T& operator [] (size_t pos) const threadgroup
    {
        return elements[pos];
    }
};

template<typename T>
inline T spvQuadSwap(T value, uint dir)
{
    return quad_shuffle_xor(value, dir + 1);
}

template<>
inline bool spvQuadSwap(bool value, uint dir)
{
    return !!quad_shuffle_xor((ushort)value, dir + 1);
}

template<uint N>
inline vec<bool, N> spvQuadSwap(vec<bool, N> value, uint dir)
{
    return (vec<bool, N>)quad_shuffle_xor((vec<ushort, N>)value, dir + 1);
}

template <typename ImageT>
void spvImageFence(ImageT img) { img.fence(); }

struct cbFI_t
{
    int2 renderSize;
    int2 displaySize;
    float2 displaySizeRcp;
    float cameraNear;
    float cameraFar;
    int2 upscalerTargetSize;
    int Mode;
    int reset;
    float4 fDeviceToViewDepth;
    float deltaTime;
    int HUDLessAttachedFactor;
    int2 distortionFieldSize;
    float2 opticalFlowScale;
    int opticalFlowBlockSize;
    uint dispatchFlags;
    int2 maxRenderSize;
    int opticalFlowHalfResMode;
    int NumInstances;
    int2 interpolationRectBase;
    int2 interpolationRectSize;
    packed_float3 debugBarColor;
    uint backBufferTransferFunction;
    float2 minMaxLuminance;
    float fTanHalfFOV;
    int _pad1;
    float2 fJitter;
    float2 fMotionVectorScale;
};

struct cbInpaintingPyramid_t
{
    uint mips;
    uint numWorkGroups;
    uint2 workGroupOffset;
};

struct FrameInterpolationRWCounters_t
{
    uint data[1];
};

constant uint3 gl_WorkGroupSize [[maybe_unused]] = uint3(256u, 1u, 1u);

[[host_name("ism_fsr_compute_game_vector_field_inpainting_pyramid")]] kernel void ism_fsr_compute_game_vector_field_inpainting_pyramid(constant cbFI_t& cbFI [[buffer(0)]], constant cbInpaintingPyramid_t& cbInpaintingPyramid [[buffer(1)]], coherent device FrameInterpolationRWCounters_t& rw_counters [[buffer(2)]], texture2d<uint> r_game_motion_vector_field_x [[texture(0)]], texture2d<uint> r_game_motion_vector_field_y [[texture(1)]], texture2d<float, access::write> rw_inpainting_pyramid0 [[texture(2)]], texture2d<float, access::write> rw_inpainting_pyramid1 [[texture(3)]], texture2d<float, access::write> rw_inpainting_pyramid2 [[texture(4)]], texture2d<float, access::write> rw_inpainting_pyramid3 [[texture(5)]], texture2d<float, access::write> rw_inpainting_pyramid4 [[texture(6)]], texture2d<float, access::read_write, memory_coherence_device> rw_inpainting_pyramid5 [[texture(7)]], texture2d<float, access::write> rw_inpainting_pyramid6 [[texture(8)]], texture2d<float, access::write> rw_inpainting_pyramid7 [[texture(9)]], texture2d<float, access::write> rw_inpainting_pyramid8 [[texture(10)]], texture2d<float, access::write> rw_inpainting_pyramid9 [[texture(11)]], texture2d<float, access::write> rw_inpainting_pyramid10 [[texture(12)]], texture2d<float, access::write> rw_inpainting_pyramid11 [[texture(13)]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]], uint gl_LocalInvocationIndex [[thread_index_in_threadgroup]])
{
    threadgroup uint spdCounter;
    threadgroup spvUnsafeArray<spvUnsafeArray<float, 16>, 16> spdIntermediateR;
    threadgroup spvUnsafeArray<spvUnsafeArray<float, 16>, 16> spdIntermediateG;
    threadgroup spvUnsafeArray<spvUnsafeArray<float, 16>, 16> spdIntermediateB;
    threadgroup spvUnsafeArray<spvUnsafeArray<float, 16>, 16> spdIntermediateA;
    uint2 _1941 = uint2(int3(gl_WorkGroupID).xy) + cbInpaintingPyramid.workGroupOffset;
    do
    {
        uint _2182;
        int _2183;
        uint _2185;
        int _2186;
        int2 _2187;
        int2 _2197;
        uint _1983 = gl_LocalInvocationIndex % 64u;
        uint _1991 = insert_bits(extract_bits(_1983, uint(2), uint(3)), _1983, uint(0), uint(1)) + (8u * ((gl_LocalInvocationIndex >> uint(6)) % 2u));
        uint _1997 = insert_bits(extract_bits(_1983, uint(3), uint(3)), extract_bits(_1983, uint(1), uint(2)), uint(0), uint(2)) + (8u * (gl_LocalInvocationIndex >> uint(7)));
        do
        {
            int2 _2180 = int2(_1941 * uint2(64u));
            _2182 = _1991 * 2u;
            _2183 = int(_2182);
            _2185 = _1997 * 2u;
            _2186 = int(_2185);
            _2187 = int2(_2183, _2186);
            int2 _2188 = _2180 + _2187;
            int2 _2192 = int2(_1941 * uint2(32u));
            int _2194 = int(_1991);
            int _2196 = int(_1997);
            _2197 = int2(_2194, _2196);
            uint2 _2200 = uint2(_2188);
            uint4 _2527 = r_game_motion_vector_field_x.read(uint2(_2188), 0);
            uint _2528 = _2527.x;
            float _2545 = float((_2528 >> 21u) & 1023u) * 0.000977517105638980865478515625;
            bool _2560 = _2545 > 0.0;
            bool _2569;
            if (_2560)
            {
                _2569 = (_2528 & 2147483648u) == 0u;
            }
            else
            {
                _2569 = _2560;
            }
            float _11828;
            if (_2569)
            {
                _11828 = 1.0 - _2545;
            }
            else
            {
                _11828 = _2545;
            }
            float _2519 = float(cbFI.displaySize.x > 0);
            float4 _2520 = float4(float2(as_type<half2>(_2528)).x, float2(as_type<half2>(r_game_motion_vector_field_y.read(uint2(_2188), 0).x)).x, _11828, float((_2528 >> 16u) & 31u) * 0.0322580635547637939453125) * _2519;
            int2 _2479 = int2(_2200 + uint2(0u, 1u));
            uint4 _2639 = r_game_motion_vector_field_x.read(uint2(_2479), 0);
            uint _2640 = _2639.x;
            float _2657 = float((_2640 >> 21u) & 1023u) * 0.000977517105638980865478515625;
            bool _2672 = _2657 > 0.0;
            bool _2681;
            if (_2672)
            {
                _2681 = (_2640 & 2147483648u) == 0u;
            }
            else
            {
                _2681 = _2672;
            }
            float _11834;
            if (_2681)
            {
                _11834 = 1.0 - _2657;
            }
            else
            {
                _11834 = _2657;
            }
            float4 _2632 = float4(float2(as_type<half2>(_2640)).x, float2(as_type<half2>(r_game_motion_vector_field_y.read(uint2(_2479), 0).x)).x, _11834, float((_2640 >> 16u) & 31u) * 0.0322580635547637939453125) * _2519;
            int2 _2483 = int2(_2200 + uint2(1u, 0u));
            uint4 _2751 = r_game_motion_vector_field_x.read(uint2(_2483), 0);
            uint _2752 = _2751.x;
            float _2769 = float((_2752 >> 21u) & 1023u) * 0.000977517105638980865478515625;
            bool _2784 = _2769 > 0.0;
            bool _2793;
            if (_2784)
            {
                _2793 = (_2752 & 2147483648u) == 0u;
            }
            else
            {
                _2793 = _2784;
            }
            float _11842;
            if (_2793)
            {
                _11842 = 1.0 - _2769;
            }
            else
            {
                _11842 = _2769;
            }
            float4 _2744 = float4(float2(as_type<half2>(_2752)).x, float2(as_type<half2>(r_game_motion_vector_field_y.read(uint2(_2483), 0).x)).x, _11842, float((_2752 >> 16u) & 31u) * 0.0322580635547637939453125) * _2519;
            int2 _2487 = int2(_2200 + uint2(1u));
            uint4 _2863 = r_game_motion_vector_field_x.read(uint2(_2487), 0);
            uint _2864 = _2863.x;
            float _2881 = float((_2864 >> 21u) & 1023u) * 0.000977517105638980865478515625;
            bool _2896 = _2881 > 0.0;
            bool _2905;
            if (_2896)
            {
                _2905 = (_2864 & 2147483648u) == 0u;
            }
            else
            {
                _2905 = _2896;
            }
            float _11852;
            if (_2905)
            {
                _11852 = 1.0 - _2881;
            }
            else
            {
                _11852 = _2881;
            }
            float4 _2856 = float4(float2(as_type<half2>(_2864)).x, float2(as_type<half2>(r_game_motion_vector_field_y.read(uint2(_2487), 0).x)).x, _11852, float((_2864 >> 16u) & 31u) * 0.0322580635547637939453125) * _2519;
            float _2954 = float(_2520.z > 0.0);
            float _2966 = float(_2632.z > 0.0);
            float _2978 = float(_2744.z > 0.0);
            float _2990 = float(_2856.z > 0.0);
            float _2998 = ((_2954 + _2966) + _2978) + _2990;
            float4 _3005 = ((((_2520 * _2954) + (_2632 * _2966)) + (_2744 * _2978)) + (_2856 * _2990)) / float4((_2998 > 0.001000000047497451305389404296875) ? _2998 : 1.0);
            rw_inpainting_pyramid0.write(_3005, uint2((_2192 + _2197)));
            int _2216 = int(_2182 + 32u);
            int2 _2221 = _2180 + int2(_2216, _2186);
            int _2228 = int(_1991 + 16u);
            uint2 _2234 = uint2(_2221);
            uint4 _3195 = r_game_motion_vector_field_x.read(uint2(_2221), 0);
            uint _3196 = _3195.x;
            float _3213 = float((_3196 >> 21u) & 1023u) * 0.000977517105638980865478515625;
            bool _3228 = _3213 > 0.0;
            bool _3237;
            if (_3228)
            {
                _3237 = (_3196 & 2147483648u) == 0u;
            }
            else
            {
                _3237 = _3228;
            }
            float _11885;
            if (_3237)
            {
                _11885 = 1.0 - _3213;
            }
            else
            {
                _11885 = _3213;
            }
            float4 _3188 = float4(float2(as_type<half2>(_3196)).x, float2(as_type<half2>(r_game_motion_vector_field_y.read(uint2(_2221), 0).x)).x, _11885, float((_3196 >> 16u) & 31u) * 0.0322580635547637939453125) * _2519;
            int2 _3147 = int2(_2234 + uint2(0u, 1u));
            uint4 _3307 = r_game_motion_vector_field_x.read(uint2(_3147), 0);
            uint _3308 = _3307.x;
            float _3325 = float((_3308 >> 21u) & 1023u) * 0.000977517105638980865478515625;
            bool _3340 = _3325 > 0.0;
            bool _3349;
            if (_3340)
            {
                _3349 = (_3308 & 2147483648u) == 0u;
            }
            else
            {
                _3349 = _3340;
            }
            float _11912;
            if (_3349)
            {
                _11912 = 1.0 - _3325;
            }
            else
            {
                _11912 = _3325;
            }
            float4 _3300 = float4(float2(as_type<half2>(_3308)).x, float2(as_type<half2>(r_game_motion_vector_field_y.read(uint2(_3147), 0).x)).x, _11912, float((_3308 >> 16u) & 31u) * 0.0322580635547637939453125) * _2519;
            int2 _3151 = int2(_2234 + uint2(1u, 0u));
            uint4 _3419 = r_game_motion_vector_field_x.read(uint2(_3151), 0);
            uint _3420 = _3419.x;
            float _3437 = float((_3420 >> 21u) & 1023u) * 0.000977517105638980865478515625;
            bool _3452 = _3437 > 0.0;
            bool _3461;
            if (_3452)
            {
                _3461 = (_3420 & 2147483648u) == 0u;
            }
            else
            {
                _3461 = _3452;
            }
            float _11941;
            if (_3461)
            {
                _11941 = 1.0 - _3437;
            }
            else
            {
                _11941 = _3437;
            }
            float4 _3412 = float4(float2(as_type<half2>(_3420)).x, float2(as_type<half2>(r_game_motion_vector_field_y.read(uint2(_3151), 0).x)).x, _11941, float((_3420 >> 16u) & 31u) * 0.0322580635547637939453125) * _2519;
            int2 _3155 = int2(_2234 + uint2(1u));
            uint4 _3531 = r_game_motion_vector_field_x.read(uint2(_3155), 0);
            uint _3532 = _3531.x;
            float _3549 = float((_3532 >> 21u) & 1023u) * 0.000977517105638980865478515625;
            bool _3564 = _3549 > 0.0;
            bool _3573;
            if (_3564)
            {
                _3573 = (_3532 & 2147483648u) == 0u;
            }
            else
            {
                _3573 = _3564;
            }
            float _11972;
            if (_3573)
            {
                _11972 = 1.0 - _3549;
            }
            else
            {
                _11972 = _3549;
            }
            float4 _3524 = float4(float2(as_type<half2>(_3532)).x, float2(as_type<half2>(r_game_motion_vector_field_y.read(uint2(_3155), 0).x)).x, _11972, float((_3532 >> 16u) & 31u) * 0.0322580635547637939453125) * _2519;
            float _3622 = float(_3188.z > 0.0);
            float _3634 = float(_3300.z > 0.0);
            float _3646 = float(_3412.z > 0.0);
            float _3658 = float(_3524.z > 0.0);
            float _3666 = ((_3622 + _3634) + _3646) + _3658;
            float4 _3673 = ((((_3188 * _3622) + (_3300 * _3634)) + (_3412 * _3646)) + (_3524 * _3658)) / float4((_3666 > 0.001000000047497451305389404296875) ? _3666 : 1.0);
            rw_inpainting_pyramid0.write(_3673, uint2((_2192 + int2(_2228, _2196))));
            int _2253 = int(_2185 + 32u);
            int2 _2255 = _2180 + int2(_2183, _2253);
            int _2264 = int(_1997 + 16u);
            uint2 _2268 = uint2(_2255);
            uint4 _3863 = r_game_motion_vector_field_x.read(uint2(_2255), 0);
            uint _3864 = _3863.x;
            float _3881 = float((_3864 >> 21u) & 1023u) * 0.000977517105638980865478515625;
            bool _3896 = _3881 > 0.0;
            bool _3905;
            if (_3896)
            {
                _3905 = (_3864 & 2147483648u) == 0u;
            }
            else
            {
                _3905 = _3896;
            }
            float _12026;
            if (_3905)
            {
                _12026 = 1.0 - _3881;
            }
            else
            {
                _12026 = _3881;
            }
            float4 _3856 = float4(float2(as_type<half2>(_3864)).x, float2(as_type<half2>(r_game_motion_vector_field_y.read(uint2(_2255), 0).x)).x, _12026, float((_3864 >> 16u) & 31u) * 0.0322580635547637939453125) * _2519;
            int2 _3815 = int2(_2268 + uint2(0u, 1u));
            uint4 _3975 = r_game_motion_vector_field_x.read(uint2(_3815), 0);
            uint _3976 = _3975.x;
            float _3993 = float((_3976 >> 21u) & 1023u) * 0.000977517105638980865478515625;
            bool _4008 = _3993 > 0.0;
            bool _4017;
            if (_4008)
            {
                _4017 = (_3976 & 2147483648u) == 0u;
            }
            else
            {
                _4017 = _4008;
            }
            float _12074;
            if (_4017)
            {
                _12074 = 1.0 - _3993;
            }
            else
            {
                _12074 = _3993;
            }
            float4 _3968 = float4(float2(as_type<half2>(_3976)).x, float2(as_type<half2>(r_game_motion_vector_field_y.read(uint2(_3815), 0).x)).x, _12074, float((_3976 >> 16u) & 31u) * 0.0322580635547637939453125) * _2519;
            int2 _3819 = int2(_2268 + uint2(1u, 0u));
            uint4 _4087 = r_game_motion_vector_field_x.read(uint2(_3819), 0);
            uint _4088 = _4087.x;
            float _4105 = float((_4088 >> 21u) & 1023u) * 0.000977517105638980865478515625;
            bool _4120 = _4105 > 0.0;
            bool _4129;
            if (_4120)
            {
                _4129 = (_4088 & 2147483648u) == 0u;
            }
            else
            {
                _4129 = _4120;
            }
            float _12124;
            if (_4129)
            {
                _12124 = 1.0 - _4105;
            }
            else
            {
                _12124 = _4105;
            }
            float4 _4080 = float4(float2(as_type<half2>(_4088)).x, float2(as_type<half2>(r_game_motion_vector_field_y.read(uint2(_3819), 0).x)).x, _12124, float((_4088 >> 16u) & 31u) * 0.0322580635547637939453125) * _2519;
            int2 _3823 = int2(_2268 + uint2(1u));
            uint4 _4199 = r_game_motion_vector_field_x.read(uint2(_3823), 0);
            uint _4200 = _4199.x;
            float _4217 = float((_4200 >> 21u) & 1023u) * 0.000977517105638980865478515625;
            bool _4232 = _4217 > 0.0;
            bool _4241;
            if (_4232)
            {
                _4241 = (_4200 & 2147483648u) == 0u;
            }
            else
            {
                _4241 = _4232;
            }
            float _12176;
            if (_4241)
            {
                _12176 = 1.0 - _4217;
            }
            else
            {
                _12176 = _4217;
            }
            float4 _4192 = float4(float2(as_type<half2>(_4200)).x, float2(as_type<half2>(r_game_motion_vector_field_y.read(uint2(_3823), 0).x)).x, _12176, float((_4200 >> 16u) & 31u) * 0.0322580635547637939453125) * _2519;
            float _4290 = float(_3856.z > 0.0);
            float _4302 = float(_3968.z > 0.0);
            float _4314 = float(_4080.z > 0.0);
            float _4326 = float(_4192.z > 0.0);
            float _4334 = ((_4290 + _4302) + _4314) + _4326;
            float4 _4341 = ((((_3856 * _4290) + (_3968 * _4302)) + (_4080 * _4314)) + (_4192 * _4326)) / float4((_4334 > 0.001000000047497451305389404296875) ? _4334 : 1.0);
            rw_inpainting_pyramid0.write(_4341, uint2((_2192 + int2(_2194, _2264))));
            int2 _2290 = _2180 + int2(_2216, _2253);
            uint2 _2304 = uint2(_2290);
            uint4 _4531 = r_game_motion_vector_field_x.read(uint2(_2290), 0);
            uint _4532 = _4531.x;
            float _4549 = float((_4532 >> 21u) & 1023u) * 0.000977517105638980865478515625;
            bool _4564 = _4549 > 0.0;
            bool _4573;
            if (_4564)
            {
                _4573 = (_4532 & 2147483648u) == 0u;
            }
            else
            {
                _4573 = _4564;
            }
            float _12251;
            if (_4573)
            {
                _12251 = 1.0 - _4549;
            }
            else
            {
                _12251 = _4549;
            }
            float4 _4524 = float4(float2(as_type<half2>(_4532)).x, float2(as_type<half2>(r_game_motion_vector_field_y.read(uint2(_2290), 0).x)).x, _12251, float((_4532 >> 16u) & 31u) * 0.0322580635547637939453125) * _2519;
            int2 _4483 = int2(_2304 + uint2(0u, 1u));
            uint4 _4643 = r_game_motion_vector_field_x.read(uint2(_4483), 0);
            uint _4644 = _4643.x;
            float _4661 = float((_4644 >> 21u) & 1023u) * 0.000977517105638980865478515625;
            bool _4676 = _4661 > 0.0;
            bool _4685;
            if (_4676)
            {
                _4685 = (_4644 & 2147483648u) == 0u;
            }
            else
            {
                _4685 = _4676;
            }
            float _12320;
            if (_4685)
            {
                _12320 = 1.0 - _4661;
            }
            else
            {
                _12320 = _4661;
            }
            float4 _4636 = float4(float2(as_type<half2>(_4644)).x, float2(as_type<half2>(r_game_motion_vector_field_y.read(uint2(_4483), 0).x)).x, _12320, float((_4644 >> 16u) & 31u) * 0.0322580635547637939453125) * _2519;
            int2 _4487 = int2(_2304 + uint2(1u, 0u));
            uint4 _4755 = r_game_motion_vector_field_x.read(uint2(_4487), 0);
            uint _4756 = _4755.x;
            float _4773 = float((_4756 >> 21u) & 1023u) * 0.000977517105638980865478515625;
            bool _4788 = _4773 > 0.0;
            bool _4797;
            if (_4788)
            {
                _4797 = (_4756 & 2147483648u) == 0u;
            }
            else
            {
                _4797 = _4788;
            }
            float _12391;
            if (_4797)
            {
                _12391 = 1.0 - _4773;
            }
            else
            {
                _12391 = _4773;
            }
            float4 _4748 = float4(float2(as_type<half2>(_4756)).x, float2(as_type<half2>(r_game_motion_vector_field_y.read(uint2(_4487), 0).x)).x, _12391, float((_4756 >> 16u) & 31u) * 0.0322580635547637939453125) * _2519;
            int2 _4491 = int2(_2304 + uint2(1u));
            uint4 _4867 = r_game_motion_vector_field_x.read(uint2(_4491), 0);
            uint _4868 = _4867.x;
            float _4885 = float((_4868 >> 21u) & 1023u) * 0.000977517105638980865478515625;
            bool _4900 = _4885 > 0.0;
            bool _4909;
            if (_4900)
            {
                _4909 = (_4868 & 2147483648u) == 0u;
            }
            else
            {
                _4909 = _4900;
            }
            float _12464;
            if (_4909)
            {
                _12464 = 1.0 - _4885;
            }
            else
            {
                _12464 = _4885;
            }
            float4 _4860 = float4(float2(as_type<half2>(_4868)).x, float2(as_type<half2>(r_game_motion_vector_field_y.read(uint2(_4491), 0).x)).x, _12464, float((_4868 >> 16u) & 31u) * 0.0322580635547637939453125) * _2519;
            float _4958 = float(_4524.z > 0.0);
            float _4970 = float(_4636.z > 0.0);
            float _4982 = float(_4748.z > 0.0);
            float _4994 = float(_4860.z > 0.0);
            float _5002 = ((_4958 + _4970) + _4982) + _4994;
            float4 _5009 = ((((_4524 * _4958) + (_4636 * _4970)) + (_4748 * _4982)) + (_4860 * _4994)) / float4((_5002 > 0.001000000047497451305389404296875) ? _5002 : 1.0);
            rw_inpainting_pyramid0.write(_5009, uint2((_2192 + int2(_2228, _2264))));
            if (cbInpaintingPyramid.mips <= 1u)
            {
                break;
            }
            float4 _5123 = spvQuadSwap(_3005, 0u);
            float4 _5125 = spvQuadSwap(_3005, 1u);
            float4 _5127 = spvQuadSwap(_3005, 2u);
            float _5144 = float(_3005.z > 0.0);
            float _5156 = float(_5123.z > 0.0);
            float _5168 = float(_5125.z > 0.0);
            float _5180 = float(_5127.z > 0.0);
            float _5188 = ((_5144 + _5156) + _5168) + _5180;
            float4 _5195 = ((((_3005 * _5144) + (_5123 * _5156)) + (_5125 * _5168)) + (_5127 * _5180)) / float4((_5188 > 0.001000000047497451305389404296875) ? _5188 : 1.0);
            float4 _5209 = spvQuadSwap(_3673, 0u);
            float4 _5211 = spvQuadSwap(_3673, 1u);
            float4 _5213 = spvQuadSwap(_3673, 2u);
            float _5230 = float(_3673.z > 0.0);
            float _5242 = float(_5209.z > 0.0);
            float _5254 = float(_5211.z > 0.0);
            float _5266 = float(_5213.z > 0.0);
            float _5274 = ((_5230 + _5242) + _5254) + _5266;
            float4 _5281 = ((((_3673 * _5230) + (_5209 * _5242)) + (_5211 * _5254)) + (_5213 * _5266)) / float4((_5274 > 0.001000000047497451305389404296875) ? _5274 : 1.0);
            float4 _5295 = spvQuadSwap(_4341, 0u);
            float4 _5297 = spvQuadSwap(_4341, 1u);
            float4 _5299 = spvQuadSwap(_4341, 2u);
            float _5316 = float(_4341.z > 0.0);
            float _5328 = float(_5295.z > 0.0);
            float _5340 = float(_5297.z > 0.0);
            float _5352 = float(_5299.z > 0.0);
            float _5360 = ((_5316 + _5328) + _5340) + _5352;
            float4 _5367 = ((((_4341 * _5316) + (_5295 * _5328)) + (_5297 * _5340)) + (_5299 * _5352)) / float4((_5360 > 0.001000000047497451305389404296875) ? _5360 : 1.0);
            float4 _5381 = spvQuadSwap(_5009, 0u);
            float4 _5383 = spvQuadSwap(_5009, 1u);
            float4 _5385 = spvQuadSwap(_5009, 2u);
            float _5402 = float(_5009.z > 0.0);
            float _5414 = float(_5381.z > 0.0);
            float _5426 = float(_5383.z > 0.0);
            float _5438 = float(_5385.z > 0.0);
            float _5446 = ((_5402 + _5414) + _5426) + _5438;
            float4 _5453 = ((((_5009 * _5402) + (_5381 * _5414)) + (_5383 * _5426)) + (_5385 * _5438)) / float4((_5446 > 0.001000000047497451305389404296875) ? _5446 : 1.0);
            if ((gl_LocalInvocationIndex % 4u) == 0u)
            {
                int2 _2340 = int2(_1941 * uint2(16u));
                uint _2342 = _1991 / 2u;
                int _2343 = int(_2342);
                uint _2345 = _1997 / 2u;
                int _2346 = int(_2345);
                rw_inpainting_pyramid1.write(_5195, uint2((_2340 + int2(_2343, _2346))));
                spdIntermediateR[_2342][_2345] = _5195.x;
                spdIntermediateG[_2342][_2345] = _5195.y;
                spdIntermediateB[_2342][_2345] = _5195.z;
                spdIntermediateA[_2342][_2345] = _5195.w;
                uint _2366 = _2342 + 8u;
                int _2367 = int(_2366);
                rw_inpainting_pyramid1.write(_5281, uint2((_2340 + int2(_2367, _2346))));
                spdIntermediateR[_2366][_2345] = _5281.x;
                spdIntermediateG[_2366][_2345] = _5281.y;
                spdIntermediateB[_2366][_2345] = _5281.z;
                spdIntermediateA[_2366][_2345] = _5281.w;
                uint _2394 = _2345 + 8u;
                int _2395 = int(_2394);
                rw_inpainting_pyramid1.write(_5367, uint2((_2340 + int2(_2343, _2395))));
                spdIntermediateR[_2342][_2394] = _5367.x;
                spdIntermediateG[_2342][_2394] = _5367.y;
                spdIntermediateB[_2342][_2394] = _5367.z;
                spdIntermediateA[_2342][_2394] = _5367.w;
                rw_inpainting_pyramid1.write(_5453, uint2((_2340 + int2(_2367, _2395))));
                spdIntermediateR[_2366][_2394] = _5453.x;
                spdIntermediateG[_2366][_2394] = _5453.y;
                spdIntermediateB[_2366][_2394] = _5453.z;
                spdIntermediateA[_2366][_2394] = _5453.w;
            }
            break;
        } while(false);
        do
        {
            if (cbInpaintingPyramid.mips <= 2u)
            {
                break;
            }
            atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture, memory_order_seq_cst, thread_scope_threadgroup);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float _6073 = spdIntermediateR[_1991][_1997];
            float _6077 = spdIntermediateG[_1991][_1997];
            float _6081 = spdIntermediateB[_1991][_1997];
            float4 _6086 = float4(_6073, _6077, _6081, spdIntermediateA[_1991][_1997]);
            float4 _6099 = spvQuadSwap(_6086, 0u);
            float4 _6101 = spvQuadSwap(_6086, 1u);
            float4 _6103 = spvQuadSwap(_6086, 2u);
            float _6120 = float(_6081 > 0.0);
            float _6132 = float(_6099.z > 0.0);
            float _6144 = float(_6101.z > 0.0);
            float _6156 = float(_6103.z > 0.0);
            float _6164 = ((_6120 + _6132) + _6144) + _6156;
            float4 _6171 = ((((_6086 * _6120) + (_6099 * _6132)) + (_6101 * _6144)) + (_6103 * _6156)) / float4((_6164 > 0.001000000047497451305389404296875) ? _6164 : 1.0);
            bool _6041 = (gl_LocalInvocationIndex % 4u) == 0u;
            if (_6041)
            {
                uint _6051 = _1997 / 2u;
                rw_inpainting_pyramid2.write(_6171, uint2((int2(_1941 * uint2(8u)) + int2(int(_1991 / 2u), int(_6051)))));
                uint _6063 = _1991 + (_6051 % 2u);
                spdIntermediateR[_6063][_1997] = _6171.x;
                spdIntermediateG[_6063][_1997] = _6171.y;
                spdIntermediateB[_6063][_1997] = _6171.z;
                spdIntermediateA[_6063][_1997] = _6171.w;
            }
            if (cbInpaintingPyramid.mips <= 3u)
            {
                break;
            }
            atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture, memory_order_seq_cst, thread_scope_threadgroup);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (gl_LocalInvocationIndex < 64u)
            {
                uint _6314 = _2182 + (_1997 % 2u);
                float _6356 = spdIntermediateR[_6314][_2185];
                float _6360 = spdIntermediateG[_6314][_2185];
                float _6364 = spdIntermediateB[_6314][_2185];
                float4 _6369 = float4(_6356, _6360, _6364, spdIntermediateA[_6314][_2185]);
                float4 _6382 = spvQuadSwap(_6369, 0u);
                float4 _6384 = spvQuadSwap(_6369, 1u);
                float4 _6386 = spvQuadSwap(_6369, 2u);
                float _6403 = float(_6364 > 0.0);
                float _6415 = float(_6382.z > 0.0);
                float _6427 = float(_6384.z > 0.0);
                float _6439 = float(_6386.z > 0.0);
                float _6447 = ((_6403 + _6415) + _6427) + _6439;
                float4 _6454 = ((((_6369 * _6403) + (_6382 * _6415)) + (_6384 * _6427)) + (_6386 * _6439)) / float4((_6447 > 0.001000000047497451305389404296875) ? _6447 : 1.0);
                if (_6041)
                {
                    uint _6332 = _1997 / 2u;
                    rw_inpainting_pyramid3.write(_6454, uint2((int2(_1941 * uint2(4u)) + int2(int(_1991 / 2u), int(_6332)))));
                    uint _6344 = _2182 + _6332;
                    spdIntermediateR[_6344][_2185] = _6454.x;
                    spdIntermediateG[_6344][_2185] = _6454.y;
                    spdIntermediateB[_6344][_2185] = _6454.z;
                    spdIntermediateA[_6344][_2185] = _6454.w;
                }
            }
            if (cbInpaintingPyramid.mips <= 4u)
            {
                break;
            }
            atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture, memory_order_seq_cst, thread_scope_threadgroup);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (gl_LocalInvocationIndex < 16u)
            {
                uint _6596 = (_1991 * 4u) + _1997;
                uint _6598 = _1997 * 4u;
                float _6635 = spdIntermediateR[_6596][_6598];
                float _6639 = spdIntermediateG[_6596][_6598];
                float _6643 = spdIntermediateB[_6596][_6598];
                float4 _6648 = float4(_6635, _6639, _6643, spdIntermediateA[_6596][_6598]);
                float4 _6661 = spvQuadSwap(_6648, 0u);
                float4 _6663 = spvQuadSwap(_6648, 1u);
                float4 _6665 = spvQuadSwap(_6648, 2u);
                float _6682 = float(_6643 > 0.0);
                float _6694 = float(_6661.z > 0.0);
                float _6706 = float(_6663.z > 0.0);
                float _6718 = float(_6665.z > 0.0);
                float _6726 = ((_6682 + _6694) + _6706) + _6718;
                float4 _6733 = ((((_6648 * _6682) + (_6661 * _6694)) + (_6663 * _6706)) + (_6665 * _6718)) / float4((_6726 > 0.001000000047497451305389404296875) ? _6726 : 1.0);
                if (_6041)
                {
                    uint _6611 = _1991 / 2u;
                    rw_inpainting_pyramid4.write(_6733, uint2((int2(_1941 * uint2(2u)) + int2(int(_6611), int(_1997 / 2u)))));
                    uint _6625 = _6611 + _1997;
                    spdIntermediateR[_6625][0u] = _6733.x;
                    spdIntermediateG[_6625][0u] = _6733.y;
                    spdIntermediateB[_6625][0u] = _6733.z;
                    spdIntermediateA[_6625][0u] = _6733.w;
                }
            }
            if (cbInpaintingPyramid.mips <= 5u)
            {
                break;
            }
            atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture, memory_order_seq_cst, thread_scope_threadgroup);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (gl_LocalInvocationIndex < 4u)
            {
                float4 _6903 = float4(spdIntermediateR[gl_LocalInvocationIndex][0u], spdIntermediateG[gl_LocalInvocationIndex][0u], spdIntermediateB[gl_LocalInvocationIndex][0u], spdIntermediateA[gl_LocalInvocationIndex][0u]);
                float4 _6916 = spvQuadSwap(_6903, 0u);
                float4 _6918 = spvQuadSwap(_6903, 1u);
                float4 _6920 = spvQuadSwap(_6903, 2u);
                float _6937 = float(spdIntermediateB[gl_LocalInvocationIndex][0u] > 0.0);
                float _6949 = float(_6916.z > 0.0);
                float _6961 = float(_6918.z > 0.0);
                float _6973 = float(_6920.z > 0.0);
                float _6981 = ((_6937 + _6949) + _6961) + _6973;
                if (_6041)
                {
                    rw_inpainting_pyramid5.write(((((_6903 * _6937) + (_6916 * _6949)) + (_6918 * _6961)) + (_6920 * _6973)) / float4((_6981 > 0.001000000047497451305389404296875) ? _6981 : 1.0), uint2(int2(_1941)));
                }
            }
            break;
        } while(false);
        if (cbInpaintingPyramid.mips <= 6u)
        {
            break;
        }
        if (gl_LocalInvocationIndex == 0u)
        {
            uint _7112 = atomic_fetch_add_explicit((coherent device atomic_uint*)&rw_counters.data[0], 1u, memory_order_relaxed);
            spdCounter = _7112;
        }
        atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture, memory_order_seq_cst, thread_scope_threadgroup);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (spdCounter != (cbInpaintingPyramid.numWorkGroups - 1u))
        {
            break;
        }
        uint _7171;
        uint _7175;
        rw_counters.data[0] = 0u;
        do
        {
            _7171 = _1991 * 4u;
            int _7173 = int(_7171);
            _7175 = _1997 * 4u;
            int _7177 = int(_7175);
            int2 _7178 = int2(_7173, _7177);
            uint2 _7189 = uint2(_7178);
            spvImageFence(rw_inpainting_pyramid5);
            float4 _7403 = rw_inpainting_pyramid5.read(uint2(_7178));
            spvImageFence(rw_inpainting_pyramid5);
            float4 _7507 = rw_inpainting_pyramid5.read(uint2(int2(_7189 + uint2(0u, 1u))));
            spvImageFence(rw_inpainting_pyramid5);
            float4 _7611 = rw_inpainting_pyramid5.read(uint2(int2(_7189 + uint2(1u, 0u))));
            spvImageFence(rw_inpainting_pyramid5);
            float4 _7715 = rw_inpainting_pyramid5.read(uint2(int2(_7189 + uint2(1u))));
            float _7779 = float(_7403.z > 0.0);
            float _7791 = float(_7507.z > 0.0);
            float _7803 = float(_7611.z > 0.0);
            float _7815 = float(_7715.z > 0.0);
            float _7823 = ((_7779 + _7791) + _7803) + _7815;
            float4 _7830 = ((((_7403 * _7779) + (_7507 * _7791)) + (_7611 * _7803)) + (_7715 * _7815)) / float4((_7823 > 0.001000000047497451305389404296875) ? _7823 : 1.0);
            rw_inpainting_pyramid6.write(_7830, uint2(_2187));
            int _7199 = int(_7171 + 2u);
            int2 _7204 = int2(_7199, _7177);
            int _7208 = int(_2182 + 1u);
            uint2 _7215 = uint2(_7204);
            spvImageFence(rw_inpainting_pyramid5);
            float4 _8039 = rw_inpainting_pyramid5.read(uint2(_7204));
            spvImageFence(rw_inpainting_pyramid5);
            float4 _8143 = rw_inpainting_pyramid5.read(uint2(int2(_7215 + uint2(0u, 1u))));
            spvImageFence(rw_inpainting_pyramid5);
            float4 _8247 = rw_inpainting_pyramid5.read(uint2(int2(_7215 + uint2(1u, 0u))));
            spvImageFence(rw_inpainting_pyramid5);
            float4 _8351 = rw_inpainting_pyramid5.read(uint2(int2(_7215 + uint2(1u))));
            float _8415 = float(_8039.z > 0.0);
            float _8427 = float(_8143.z > 0.0);
            float _8439 = float(_8247.z > 0.0);
            float _8451 = float(_8351.z > 0.0);
            float _8459 = ((_8415 + _8427) + _8439) + _8451;
            float4 _8466 = ((((_8039 * _8415) + (_8143 * _8427)) + (_8247 * _8439)) + (_8351 * _8451)) / float4((_8459 > 0.001000000047497451305389404296875) ? _8459 : 1.0);
            rw_inpainting_pyramid6.write(_8466, uint2(int2(_7208, _2186)));
            int _7229 = int(_7175 + 2u);
            int2 _7230 = int2(_7173, _7229);
            int _7238 = int(_2185 + 1u);
            uint2 _7241 = uint2(_7230);
            spvImageFence(rw_inpainting_pyramid5);
            float4 _8675 = rw_inpainting_pyramid5.read(uint2(_7230));
            spvImageFence(rw_inpainting_pyramid5);
            float4 _8779 = rw_inpainting_pyramid5.read(uint2(int2(_7241 + uint2(0u, 1u))));
            spvImageFence(rw_inpainting_pyramid5);
            float4 _8883 = rw_inpainting_pyramid5.read(uint2(int2(_7241 + uint2(1u, 0u))));
            spvImageFence(rw_inpainting_pyramid5);
            float4 _8987 = rw_inpainting_pyramid5.read(uint2(int2(_7241 + uint2(1u))));
            float _9051 = float(_8675.z > 0.0);
            float _9063 = float(_8779.z > 0.0);
            float _9075 = float(_8883.z > 0.0);
            float _9087 = float(_8987.z > 0.0);
            float _9095 = ((_9051 + _9063) + _9075) + _9087;
            float4 _9102 = ((((_8675 * _9051) + (_8779 * _9063)) + (_8883 * _9075)) + (_8987 * _9087)) / float4((_9095 > 0.001000000047497451305389404296875) ? _9095 : 1.0);
            rw_inpainting_pyramid6.write(_9102, uint2(int2(_2183, _7238)));
            int2 _7256 = int2(_7199, _7229);
            uint2 _7267 = uint2(_7256);
            spvImageFence(rw_inpainting_pyramid5);
            float4 _9311 = rw_inpainting_pyramid5.read(uint2(_7256));
            spvImageFence(rw_inpainting_pyramid5);
            float4 _9415 = rw_inpainting_pyramid5.read(uint2(int2(_7267 + uint2(0u, 1u))));
            spvImageFence(rw_inpainting_pyramid5);
            float4 _9519 = rw_inpainting_pyramid5.read(uint2(int2(_7267 + uint2(1u, 0u))));
            spvImageFence(rw_inpainting_pyramid5);
            float4 _9623 = rw_inpainting_pyramid5.read(uint2(int2(_7267 + uint2(1u))));
            float _9687 = float(_9311.z > 0.0);
            float _9699 = float(_9415.z > 0.0);
            float _9711 = float(_9519.z > 0.0);
            float _9723 = float(_9623.z > 0.0);
            float _9731 = ((_9687 + _9699) + _9711) + _9723;
            float4 _9738 = ((((_9311 * _9687) + (_9415 * _9699)) + (_9519 * _9711)) + (_9623 * _9723)) / float4((_9731 > 0.001000000047497451305389404296875) ? _9731 : 1.0);
            rw_inpainting_pyramid6.write(_9738, uint2(int2(_7208, _7238)));
            if (cbInpaintingPyramid.mips <= 7u)
            {
                break;
            }
            float _9851 = float(_7830.z > 0.0);
            float _9863 = float(_8466.z > 0.0);
            float _9875 = float(_9102.z > 0.0);
            float _9887 = float(_9738.z > 0.0);
            float _9895 = ((_9851 + _9863) + _9875) + _9887;
            float4 _9902 = ((((_7830 * _9851) + (_8466 * _9863)) + (_9102 * _9875)) + (_9738 * _9887)) / float4((_9895 > 0.001000000047497451305389404296875) ? _9895 : 1.0);
            rw_inpainting_pyramid7.write(_9902, uint2(_2197));
            spdIntermediateR[_1991][_1997] = _9902.x;
            spdIntermediateG[_1991][_1997] = _9902.y;
            spdIntermediateB[_1991][_1997] = _9902.z;
            spdIntermediateA[_1991][_1997] = _9902.w;
            break;
        } while(false);
        do
        {
            if (cbInpaintingPyramid.mips <= 8u)
            {
                break;
            }
            atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture, memory_order_seq_cst, thread_scope_threadgroup);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float _10159 = spdIntermediateR[_1991][_1997];
            float _10163 = spdIntermediateG[_1991][_1997];
            float _10167 = spdIntermediateB[_1991][_1997];
            float4 _10172 = float4(_10159, _10163, _10167, spdIntermediateA[_1991][_1997]);
            float4 _10185 = spvQuadSwap(_10172, 0u);
            float4 _10187 = spvQuadSwap(_10172, 1u);
            float4 _10189 = spvQuadSwap(_10172, 2u);
            float _10206 = float(_10167 > 0.0);
            float _10218 = float(_10185.z > 0.0);
            float _10230 = float(_10187.z > 0.0);
            float _10242 = float(_10189.z > 0.0);
            float _10250 = ((_10206 + _10218) + _10230) + _10242;
            float4 _10257 = ((((_10172 * _10206) + (_10185 * _10218)) + (_10187 * _10230)) + (_10189 * _10242)) / float4((_10250 > 0.001000000047497451305389404296875) ? _10250 : 1.0);
            bool _10127 = (gl_LocalInvocationIndex % 4u) == 0u;
            if (_10127)
            {
                uint _10137 = _1997 / 2u;
                rw_inpainting_pyramid8.write(_10257, uint2(int2(int(_1991 / 2u), int(_10137))));
                uint _10149 = _1991 + (_10137 % 2u);
                spdIntermediateR[_10149][_1997] = _10257.x;
                spdIntermediateG[_10149][_1997] = _10257.y;
                spdIntermediateB[_10149][_1997] = _10257.z;
                spdIntermediateA[_10149][_1997] = _10257.w;
            }
            if (cbInpaintingPyramid.mips <= 9u)
            {
                break;
            }
            atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture, memory_order_seq_cst, thread_scope_threadgroup);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (gl_LocalInvocationIndex < 64u)
            {
                uint _10400 = _2182 + (_1997 % 2u);
                float _10442 = spdIntermediateR[_10400][_2185];
                float _10446 = spdIntermediateG[_10400][_2185];
                float _10450 = spdIntermediateB[_10400][_2185];
                float4 _10455 = float4(_10442, _10446, _10450, spdIntermediateA[_10400][_2185]);
                float4 _10468 = spvQuadSwap(_10455, 0u);
                float4 _10470 = spvQuadSwap(_10455, 1u);
                float4 _10472 = spvQuadSwap(_10455, 2u);
                float _10489 = float(_10450 > 0.0);
                float _10501 = float(_10468.z > 0.0);
                float _10513 = float(_10470.z > 0.0);
                float _10525 = float(_10472.z > 0.0);
                float _10533 = ((_10489 + _10501) + _10513) + _10525;
                float4 _10540 = ((((_10455 * _10489) + (_10468 * _10501)) + (_10470 * _10513)) + (_10472 * _10525)) / float4((_10533 > 0.001000000047497451305389404296875) ? _10533 : 1.0);
                if (_10127)
                {
                    uint _10418 = _1997 / 2u;
                    rw_inpainting_pyramid9.write(_10540, uint2(int2(int(_1991 / 2u), int(_10418))));
                    uint _10430 = _2182 + _10418;
                    spdIntermediateR[_10430][_2185] = _10540.x;
                    spdIntermediateG[_10430][_2185] = _10540.y;
                    spdIntermediateB[_10430][_2185] = _10540.z;
                    spdIntermediateA[_10430][_2185] = _10540.w;
                }
            }
            if (cbInpaintingPyramid.mips <= 10u)
            {
                break;
            }
            atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture, memory_order_seq_cst, thread_scope_threadgroup);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (gl_LocalInvocationIndex < 16u)
            {
                uint _10682 = _7171 + _1997;
                float _10721 = spdIntermediateR[_10682][_7175];
                float _10725 = spdIntermediateG[_10682][_7175];
                float _10729 = spdIntermediateB[_10682][_7175];
                float4 _10734 = float4(_10721, _10725, _10729, spdIntermediateA[_10682][_7175]);
                float4 _10747 = spvQuadSwap(_10734, 0u);
                float4 _10749 = spvQuadSwap(_10734, 1u);
                float4 _10751 = spvQuadSwap(_10734, 2u);
                float _10768 = float(_10729 > 0.0);
                float _10780 = float(_10747.z > 0.0);
                float _10792 = float(_10749.z > 0.0);
                float _10804 = float(_10751.z > 0.0);
                float _10812 = ((_10768 + _10780) + _10792) + _10804;
                float4 _10819 = ((((_10734 * _10768) + (_10747 * _10780)) + (_10749 * _10792)) + (_10751 * _10804)) / float4((_10812 > 0.001000000047497451305389404296875) ? _10812 : 1.0);
                if (_10127)
                {
                    uint _10697 = _1991 / 2u;
                    rw_inpainting_pyramid10.write(_10819, uint2(int2(int(_10697), int(_1997 / 2u))));
                    uint _10711 = _10697 + _1997;
                    spdIntermediateR[_10711][0u] = _10819.x;
                    spdIntermediateG[_10711][0u] = _10819.y;
                    spdIntermediateB[_10711][0u] = _10819.z;
                    spdIntermediateA[_10711][0u] = _10819.w;
                }
            }
            if (cbInpaintingPyramid.mips <= 11u)
            {
                break;
            }
            atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture, memory_order_seq_cst, thread_scope_threadgroup);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (gl_LocalInvocationIndex < 4u)
            {
                float4 _10989 = float4(spdIntermediateR[gl_LocalInvocationIndex][0u], spdIntermediateG[gl_LocalInvocationIndex][0u], spdIntermediateB[gl_LocalInvocationIndex][0u], spdIntermediateA[gl_LocalInvocationIndex][0u]);
                float4 _11002 = spvQuadSwap(_10989, 0u);
                float4 _11004 = spvQuadSwap(_10989, 1u);
                float4 _11006 = spvQuadSwap(_10989, 2u);
                float _11023 = float(spdIntermediateB[gl_LocalInvocationIndex][0u] > 0.0);
                float _11035 = float(_11002.z > 0.0);
                float _11047 = float(_11004.z > 0.0);
                float _11059 = float(_11006.z > 0.0);
                float _11067 = ((_11023 + _11035) + _11047) + _11059;
                if (_10127)
                {
                    rw_inpainting_pyramid11.write(((((_10989 * _11023) + (_11002 * _11035)) + (_11004 * _11047)) + (_11006 * _11059)) / float4((_11067 > 0.001000000047497451305389404296875) ? _11067 : 1.0), uint2(int2(0)));
                }
            }
            break;
        } while(false);
        break;
    } while(false);
}


}

// ===== AMD FSR compute_inpainting_pyramid =====
namespace ism_compute_inpainting_pyramid {
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wmissing-braces"


using namespace metal;

template<typename T, size_t Num>
struct spvUnsafeArray
{
    T elements[Num ? Num : 1];
    
    thread T& operator [] (size_t pos) thread
    {
        return elements[pos];
    }
    constexpr const thread T& operator [] (size_t pos) const thread
    {
        return elements[pos];
    }
    
    device T& operator [] (size_t pos) device
    {
        return elements[pos];
    }
    constexpr const device T& operator [] (size_t pos) const device
    {
        return elements[pos];
    }
    
    constexpr const constant T& operator [] (size_t pos) const constant
    {
        return elements[pos];
    }
    
    threadgroup T& operator [] (size_t pos) threadgroup
    {
        return elements[pos];
    }
    constexpr const threadgroup T& operator [] (size_t pos) const threadgroup
    {
        return elements[pos];
    }
};

template<typename T>
inline T spvQuadSwap(T value, uint dir)
{
    return quad_shuffle_xor(value, dir + 1);
}

template<>
inline bool spvQuadSwap(bool value, uint dir)
{
    return !!quad_shuffle_xor((ushort)value, dir + 1);
}

template<uint N>
inline vec<bool, N> spvQuadSwap(vec<bool, N> value, uint dir)
{
    return (vec<bool, N>)quad_shuffle_xor((vec<ushort, N>)value, dir + 1);
}

template <typename ImageT>
void spvImageFence(ImageT img) { img.fence(); }

struct cbFI_t
{
    int2 renderSize;
    int2 displaySize;
    float2 displaySizeRcp;
    float cameraNear;
    float cameraFar;
    int2 upscalerTargetSize;
    int Mode;
    int reset;
    float4 fDeviceToViewDepth;
    float deltaTime;
    int HUDLessAttachedFactor;
    int2 distortionFieldSize;
    float2 opticalFlowScale;
    int opticalFlowBlockSize;
    uint dispatchFlags;
    int2 maxRenderSize;
    int opticalFlowHalfResMode;
    int NumInstances;
    int2 interpolationRectBase;
    int2 interpolationRectSize;
    packed_float3 debugBarColor;
    uint backBufferTransferFunction;
    float2 minMaxLuminance;
    float fTanHalfFOV;
    int _pad1;
    float2 fJitter;
    float2 fMotionVectorScale;
};

struct cbInpaintingPyramid_t
{
    uint mips;
    uint numWorkGroups;
    uint2 workGroupOffset;
};

struct FrameInterpolationRWCounters_t
{
    uint data[1];
};

constant uint3 gl_WorkGroupSize [[maybe_unused]] = uint3(256u, 1u, 1u);

[[host_name("ism_fsr_compute_inpainting_pyramid")]] kernel void ism_fsr_compute_inpainting_pyramid(constant cbFI_t& cbFI [[buffer(0)]], constant cbInpaintingPyramid_t& cbInpaintingPyramid [[buffer(1)]], coherent device FrameInterpolationRWCounters_t& rw_counters [[buffer(2)]], texture2d<float> r_output [[texture(0)]], texture2d<float, access::write> rw_inpainting_pyramid0 [[texture(1)]], texture2d<float, access::write> rw_inpainting_pyramid1 [[texture(2)]], texture2d<float, access::write> rw_inpainting_pyramid2 [[texture(3)]], texture2d<float, access::write> rw_inpainting_pyramid3 [[texture(4)]], texture2d<float, access::write> rw_inpainting_pyramid4 [[texture(5)]], texture2d<float, access::read_write, memory_coherence_device> rw_inpainting_pyramid5 [[texture(6)]], texture2d<float, access::write> rw_inpainting_pyramid6 [[texture(7)]], texture2d<float, access::write> rw_inpainting_pyramid7 [[texture(8)]], texture2d<float, access::write> rw_inpainting_pyramid8 [[texture(9)]], texture2d<float, access::write> rw_inpainting_pyramid9 [[texture(10)]], texture2d<float, access::write> rw_inpainting_pyramid10 [[texture(11)]], texture2d<float, access::write> rw_inpainting_pyramid11 [[texture(12)]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]], uint gl_LocalInvocationIndex [[thread_index_in_threadgroup]])
{
    threadgroup uint spdCounter;
    threadgroup spvUnsafeArray<spvUnsafeArray<float, 16>, 16> spdIntermediateR;
    threadgroup spvUnsafeArray<spvUnsafeArray<float, 16>, 16> spdIntermediateG;
    threadgroup spvUnsafeArray<spvUnsafeArray<float, 16>, 16> spdIntermediateB;
    threadgroup spvUnsafeArray<spvUnsafeArray<float, 16>, 16> spdIntermediateA;
    uint2 _1894 = uint2(int3(gl_WorkGroupID).xy) + cbInpaintingPyramid.workGroupOffset;
    do
    {
        uint _2135;
        int _2136;
        uint _2138;
        int _2139;
        int2 _2140;
        int2 _2150;
        uint _1936 = gl_LocalInvocationIndex % 64u;
        uint _1944 = insert_bits(extract_bits(_1936, uint(2), uint(3)), _1936, uint(0), uint(1)) + (8u * ((gl_LocalInvocationIndex >> uint(6)) % 2u));
        uint _1950 = insert_bits(extract_bits(_1936, uint(3), uint(3)), extract_bits(_1936, uint(1), uint(2)), uint(0), uint(2)) + (8u * (gl_LocalInvocationIndex >> uint(7)));
        do
        {
            int2 _2133 = int2(_1894 * uint2(64u));
            _2135 = _1944 * 2u;
            _2136 = int(_2135);
            _2138 = _1950 * 2u;
            _2139 = int(_2138);
            _2140 = int2(_2136, _2139);
            int2 _2141 = _2133 + _2140;
            int2 _2145 = int2(_1894 * uint2(32u));
            int _2147 = int(_1944);
            int _2149 = int(_1950);
            _2150 = int2(_2147, _2149);
            uint2 _2153 = uint2(_2141);
            float _2458 = float(cbFI.displaySize.x > 0);
            float4 _2459 = r_output.read(uint2(_2141), 0) * _2458;
            _2459.w = fast::clamp(1.0 - _2459.w, 0.0, 1.0);
            int _2466 = _2141.x;
            bool _2469 = _2466 < cbFI.interpolationRectBase.x;
            bool _2481;
            if (!_2469)
            {
                _2481 = _2466 >= (cbFI.interpolationRectSize.x + cbFI.interpolationRectBase.x);
            }
            else
            {
                _2481 = _2469;
            }
            bool _2490;
            if (!_2481)
            {
                _2490 = _2141.y < cbFI.interpolationRectBase.y;
            }
            else
            {
                _2490 = _2481;
            }
            bool _2502;
            if (!_2490)
            {
                _2502 = _2141.y >= (cbFI.interpolationRectSize.y + cbFI.interpolationRectBase.y);
            }
            else
            {
                _2502 = _2490;
            }
            float4 _11133;
            if (_2502)
            {
                float4 _10698 = _2459;
                _10698.w = 0.0;
                _11133 = _10698;
            }
            else
            {
                _11133 = _2459;
            }
            int2 _2432 = int2(_2153 + uint2(0u, 1u));
            float4 _2555 = r_output.read(uint2(_2432), 0) * _2458;
            _2555.w = fast::clamp(1.0 - _2555.w, 0.0, 1.0);
            int _2562 = _2432.x;
            bool _2565 = _2562 < cbFI.interpolationRectBase.x;
            bool _2577;
            if (!_2565)
            {
                _2577 = _2562 >= (cbFI.interpolationRectSize.x + cbFI.interpolationRectBase.x);
            }
            else
            {
                _2577 = _2565;
            }
            bool _2586;
            if (!_2577)
            {
                _2586 = _2432.y < cbFI.interpolationRectBase.y;
            }
            else
            {
                _2586 = _2577;
            }
            bool _2598;
            if (!_2586)
            {
                _2598 = _2432.y >= (cbFI.interpolationRectSize.y + cbFI.interpolationRectBase.y);
            }
            else
            {
                _2598 = _2586;
            }
            float4 _11137;
            if (_2598)
            {
                float4 _10707 = _2555;
                _10707.w = 0.0;
                _11137 = _10707;
            }
            else
            {
                _11137 = _2555;
            }
            int2 _2436 = int2(_2153 + uint2(1u, 0u));
            float4 _2651 = r_output.read(uint2(_2436), 0) * _2458;
            _2651.w = fast::clamp(1.0 - _2651.w, 0.0, 1.0);
            int _2658 = _2436.x;
            bool _2661 = _2658 < cbFI.interpolationRectBase.x;
            bool _2673;
            if (!_2661)
            {
                _2673 = _2658 >= (cbFI.interpolationRectSize.x + cbFI.interpolationRectBase.x);
            }
            else
            {
                _2673 = _2661;
            }
            bool _2682;
            if (!_2673)
            {
                _2682 = _2436.y < cbFI.interpolationRectBase.y;
            }
            else
            {
                _2682 = _2673;
            }
            bool _2694;
            if (!_2682)
            {
                _2694 = _2436.y >= (cbFI.interpolationRectSize.y + cbFI.interpolationRectBase.y);
            }
            else
            {
                _2694 = _2682;
            }
            float4 _11141;
            if (_2694)
            {
                float4 _10716 = _2651;
                _10716.w = 0.0;
                _11141 = _10716;
            }
            else
            {
                _11141 = _2651;
            }
            int2 _2440 = int2(_2153 + uint2(1u));
            float4 _2747 = r_output.read(uint2(_2440), 0) * _2458;
            _2747.w = fast::clamp(1.0 - _2747.w, 0.0, 1.0);
            int _2754 = _2440.x;
            bool _2757 = _2754 < cbFI.interpolationRectBase.x;
            bool _2769;
            if (!_2757)
            {
                _2769 = _2754 >= (cbFI.interpolationRectSize.x + cbFI.interpolationRectBase.x);
            }
            else
            {
                _2769 = _2757;
            }
            bool _2778;
            if (!_2769)
            {
                _2778 = _2440.y < cbFI.interpolationRectBase.y;
            }
            else
            {
                _2778 = _2769;
            }
            bool _2790;
            if (!_2778)
            {
                _2790 = _2440.y >= (cbFI.interpolationRectSize.y + cbFI.interpolationRectBase.y);
            }
            else
            {
                _2790 = _2778;
            }
            float4 _11145;
            if (_2790)
            {
                float4 _10725 = _2747;
                _10725.w = 0.0;
                _11145 = _10725;
            }
            else
            {
                _11145 = _2747;
            }
            float4 _11146;
            do
            {
                float _2858 = ((_11133.w + _11137.w) + _11141.w) + _11145.w;
                if (_2858 == 0.0)
                {
                    _11146 = float4(0.0);
                    break;
                }
                _11146 = ((((_11133 * _11133.w) + (_11137 * _11137.w)) + (_11141 * _11141.w)) + (_11145 * _11145.w)) / float4(_2858);
                break;
            } while(false);
            rw_inpainting_pyramid0.write(_11146, uint2((_2145 + _2150)));
            int _2169 = int(_2135 + 32u);
            int2 _2174 = _2133 + int2(_2169, _2139);
            int _2181 = int(_1944 + 16u);
            uint2 _2187 = uint2(_2174);
            float4 _3054 = r_output.read(uint2(_2174), 0) * _2458;
            _3054.w = fast::clamp(1.0 - _3054.w, 0.0, 1.0);
            int _3061 = _2174.x;
            bool _3064 = _3061 < cbFI.interpolationRectBase.x;
            bool _3076;
            if (!_3064)
            {
                _3076 = _3061 >= (cbFI.interpolationRectSize.x + cbFI.interpolationRectBase.x);
            }
            else
            {
                _3076 = _3064;
            }
            bool _3085;
            if (!_3076)
            {
                _3085 = _2174.y < cbFI.interpolationRectBase.y;
            }
            else
            {
                _3085 = _3076;
            }
            bool _3097;
            if (!_3085)
            {
                _3097 = _2174.y >= (cbFI.interpolationRectSize.y + cbFI.interpolationRectBase.y);
            }
            else
            {
                _3097 = _3085;
            }
            float4 _11167;
            if (_3097)
            {
                float4 _10746 = _3054;
                _10746.w = 0.0;
                _11167 = _10746;
            }
            else
            {
                _11167 = _3054;
            }
            int2 _3027 = int2(_2187 + uint2(0u, 1u));
            float4 _3150 = r_output.read(uint2(_3027), 0) * _2458;
            _3150.w = fast::clamp(1.0 - _3150.w, 0.0, 1.0);
            int _3157 = _3027.x;
            bool _3160 = _3157 < cbFI.interpolationRectBase.x;
            bool _3172;
            if (!_3160)
            {
                _3172 = _3157 >= (cbFI.interpolationRectSize.x + cbFI.interpolationRectBase.x);
            }
            else
            {
                _3172 = _3160;
            }
            bool _3181;
            if (!_3172)
            {
                _3181 = _3027.y < cbFI.interpolationRectBase.y;
            }
            else
            {
                _3181 = _3172;
            }
            bool _3193;
            if (!_3181)
            {
                _3193 = _3027.y >= (cbFI.interpolationRectSize.y + cbFI.interpolationRectBase.y);
            }
            else
            {
                _3193 = _3181;
            }
            float4 _11171;
            if (_3193)
            {
                float4 _10755 = _3150;
                _10755.w = 0.0;
                _11171 = _10755;
            }
            else
            {
                _11171 = _3150;
            }
            int2 _3031 = int2(_2187 + uint2(1u, 0u));
            float4 _3246 = r_output.read(uint2(_3031), 0) * _2458;
            _3246.w = fast::clamp(1.0 - _3246.w, 0.0, 1.0);
            int _3253 = _3031.x;
            bool _3256 = _3253 < cbFI.interpolationRectBase.x;
            bool _3268;
            if (!_3256)
            {
                _3268 = _3253 >= (cbFI.interpolationRectSize.x + cbFI.interpolationRectBase.x);
            }
            else
            {
                _3268 = _3256;
            }
            bool _3277;
            if (!_3268)
            {
                _3277 = _3031.y < cbFI.interpolationRectBase.y;
            }
            else
            {
                _3277 = _3268;
            }
            bool _3289;
            if (!_3277)
            {
                _3289 = _3031.y >= (cbFI.interpolationRectSize.y + cbFI.interpolationRectBase.y);
            }
            else
            {
                _3289 = _3277;
            }
            float4 _11175;
            if (_3289)
            {
                float4 _10764 = _3246;
                _10764.w = 0.0;
                _11175 = _10764;
            }
            else
            {
                _11175 = _3246;
            }
            int2 _3035 = int2(_2187 + uint2(1u));
            float4 _3342 = r_output.read(uint2(_3035), 0) * _2458;
            _3342.w = fast::clamp(1.0 - _3342.w, 0.0, 1.0);
            int _3349 = _3035.x;
            bool _3352 = _3349 < cbFI.interpolationRectBase.x;
            bool _3364;
            if (!_3352)
            {
                _3364 = _3349 >= (cbFI.interpolationRectSize.x + cbFI.interpolationRectBase.x);
            }
            else
            {
                _3364 = _3352;
            }
            bool _3373;
            if (!_3364)
            {
                _3373 = _3035.y < cbFI.interpolationRectBase.y;
            }
            else
            {
                _3373 = _3364;
            }
            bool _3385;
            if (!_3373)
            {
                _3385 = _3035.y >= (cbFI.interpolationRectSize.y + cbFI.interpolationRectBase.y);
            }
            else
            {
                _3385 = _3373;
            }
            float4 _11179;
            if (_3385)
            {
                float4 _10773 = _3342;
                _10773.w = 0.0;
                _11179 = _10773;
            }
            else
            {
                _11179 = _3342;
            }
            float4 _11180;
            do
            {
                float _3453 = ((_11167.w + _11171.w) + _11175.w) + _11179.w;
                if (_3453 == 0.0)
                {
                    _11180 = float4(0.0);
                    break;
                }
                _11180 = ((((_11167 * _11167.w) + (_11171 * _11171.w)) + (_11175 * _11175.w)) + (_11179 * _11179.w)) / float4(_3453);
                break;
            } while(false);
            rw_inpainting_pyramid0.write(_11180, uint2((_2145 + int2(_2181, _2149))));
            int _2206 = int(_2138 + 32u);
            int2 _2208 = _2133 + int2(_2136, _2206);
            int _2217 = int(_1950 + 16u);
            uint2 _2221 = uint2(_2208);
            float4 _3649 = r_output.read(uint2(_2208), 0) * _2458;
            _3649.w = fast::clamp(1.0 - _3649.w, 0.0, 1.0);
            int _3656 = _2208.x;
            bool _3659 = _3656 < cbFI.interpolationRectBase.x;
            bool _3671;
            if (!_3659)
            {
                _3671 = _3656 >= (cbFI.interpolationRectSize.x + cbFI.interpolationRectBase.x);
            }
            else
            {
                _3671 = _3659;
            }
            bool _3680;
            if (!_3671)
            {
                _3680 = _2208.y < cbFI.interpolationRectBase.y;
            }
            else
            {
                _3680 = _3671;
            }
            bool _3692;
            if (!_3680)
            {
                _3692 = _2208.y >= (cbFI.interpolationRectSize.y + cbFI.interpolationRectBase.y);
            }
            else
            {
                _3692 = _3680;
            }
            float4 _11201;
            if (_3692)
            {
                float4 _10794 = _3649;
                _10794.w = 0.0;
                _11201 = _10794;
            }
            else
            {
                _11201 = _3649;
            }
            int2 _3622 = int2(_2221 + uint2(0u, 1u));
            float4 _3745 = r_output.read(uint2(_3622), 0) * _2458;
            _3745.w = fast::clamp(1.0 - _3745.w, 0.0, 1.0);
            int _3752 = _3622.x;
            bool _3755 = _3752 < cbFI.interpolationRectBase.x;
            bool _3767;
            if (!_3755)
            {
                _3767 = _3752 >= (cbFI.interpolationRectSize.x + cbFI.interpolationRectBase.x);
            }
            else
            {
                _3767 = _3755;
            }
            bool _3776;
            if (!_3767)
            {
                _3776 = _3622.y < cbFI.interpolationRectBase.y;
            }
            else
            {
                _3776 = _3767;
            }
            bool _3788;
            if (!_3776)
            {
                _3788 = _3622.y >= (cbFI.interpolationRectSize.y + cbFI.interpolationRectBase.y);
            }
            else
            {
                _3788 = _3776;
            }
            float4 _11205;
            if (_3788)
            {
                float4 _10803 = _3745;
                _10803.w = 0.0;
                _11205 = _10803;
            }
            else
            {
                _11205 = _3745;
            }
            int2 _3626 = int2(_2221 + uint2(1u, 0u));
            float4 _3841 = r_output.read(uint2(_3626), 0) * _2458;
            _3841.w = fast::clamp(1.0 - _3841.w, 0.0, 1.0);
            int _3848 = _3626.x;
            bool _3851 = _3848 < cbFI.interpolationRectBase.x;
            bool _3863;
            if (!_3851)
            {
                _3863 = _3848 >= (cbFI.interpolationRectSize.x + cbFI.interpolationRectBase.x);
            }
            else
            {
                _3863 = _3851;
            }
            bool _3872;
            if (!_3863)
            {
                _3872 = _3626.y < cbFI.interpolationRectBase.y;
            }
            else
            {
                _3872 = _3863;
            }
            bool _3884;
            if (!_3872)
            {
                _3884 = _3626.y >= (cbFI.interpolationRectSize.y + cbFI.interpolationRectBase.y);
            }
            else
            {
                _3884 = _3872;
            }
            float4 _11209;
            if (_3884)
            {
                float4 _10812 = _3841;
                _10812.w = 0.0;
                _11209 = _10812;
            }
            else
            {
                _11209 = _3841;
            }
            int2 _3630 = int2(_2221 + uint2(1u));
            float4 _3937 = r_output.read(uint2(_3630), 0) * _2458;
            _3937.w = fast::clamp(1.0 - _3937.w, 0.0, 1.0);
            int _3944 = _3630.x;
            bool _3947 = _3944 < cbFI.interpolationRectBase.x;
            bool _3959;
            if (!_3947)
            {
                _3959 = _3944 >= (cbFI.interpolationRectSize.x + cbFI.interpolationRectBase.x);
            }
            else
            {
                _3959 = _3947;
            }
            bool _3968;
            if (!_3959)
            {
                _3968 = _3630.y < cbFI.interpolationRectBase.y;
            }
            else
            {
                _3968 = _3959;
            }
            bool _3980;
            if (!_3968)
            {
                _3980 = _3630.y >= (cbFI.interpolationRectSize.y + cbFI.interpolationRectBase.y);
            }
            else
            {
                _3980 = _3968;
            }
            float4 _11213;
            if (_3980)
            {
                float4 _10821 = _3937;
                _10821.w = 0.0;
                _11213 = _10821;
            }
            else
            {
                _11213 = _3937;
            }
            float4 _11214;
            do
            {
                float _4048 = ((_11201.w + _11205.w) + _11209.w) + _11213.w;
                if (_4048 == 0.0)
                {
                    _11214 = float4(0.0);
                    break;
                }
                _11214 = ((((_11201 * _11201.w) + (_11205 * _11205.w)) + (_11209 * _11209.w)) + (_11213 * _11213.w)) / float4(_4048);
                break;
            } while(false);
            rw_inpainting_pyramid0.write(_11214, uint2((_2145 + int2(_2147, _2217))));
            int2 _2243 = _2133 + int2(_2169, _2206);
            uint2 _2257 = uint2(_2243);
            float4 _4244 = r_output.read(uint2(_2243), 0) * _2458;
            _4244.w = fast::clamp(1.0 - _4244.w, 0.0, 1.0);
            int _4251 = _2243.x;
            bool _4254 = _4251 < cbFI.interpolationRectBase.x;
            bool _4266;
            if (!_4254)
            {
                _4266 = _4251 >= (cbFI.interpolationRectSize.x + cbFI.interpolationRectBase.x);
            }
            else
            {
                _4266 = _4254;
            }
            bool _4275;
            if (!_4266)
            {
                _4275 = _2243.y < cbFI.interpolationRectBase.y;
            }
            else
            {
                _4275 = _4266;
            }
            bool _4287;
            if (!_4275)
            {
                _4287 = _2243.y >= (cbFI.interpolationRectSize.y + cbFI.interpolationRectBase.y);
            }
            else
            {
                _4287 = _4275;
            }
            float4 _11235;
            if (_4287)
            {
                float4 _10842 = _4244;
                _10842.w = 0.0;
                _11235 = _10842;
            }
            else
            {
                _11235 = _4244;
            }
            int2 _4217 = int2(_2257 + uint2(0u, 1u));
            float4 _4340 = r_output.read(uint2(_4217), 0) * _2458;
            _4340.w = fast::clamp(1.0 - _4340.w, 0.0, 1.0);
            int _4347 = _4217.x;
            bool _4350 = _4347 < cbFI.interpolationRectBase.x;
            bool _4362;
            if (!_4350)
            {
                _4362 = _4347 >= (cbFI.interpolationRectSize.x + cbFI.interpolationRectBase.x);
            }
            else
            {
                _4362 = _4350;
            }
            bool _4371;
            if (!_4362)
            {
                _4371 = _4217.y < cbFI.interpolationRectBase.y;
            }
            else
            {
                _4371 = _4362;
            }
            bool _4383;
            if (!_4371)
            {
                _4383 = _4217.y >= (cbFI.interpolationRectSize.y + cbFI.interpolationRectBase.y);
            }
            else
            {
                _4383 = _4371;
            }
            float4 _11239;
            if (_4383)
            {
                float4 _10851 = _4340;
                _10851.w = 0.0;
                _11239 = _10851;
            }
            else
            {
                _11239 = _4340;
            }
            int2 _4221 = int2(_2257 + uint2(1u, 0u));
            float4 _4436 = r_output.read(uint2(_4221), 0) * _2458;
            _4436.w = fast::clamp(1.0 - _4436.w, 0.0, 1.0);
            int _4443 = _4221.x;
            bool _4446 = _4443 < cbFI.interpolationRectBase.x;
            bool _4458;
            if (!_4446)
            {
                _4458 = _4443 >= (cbFI.interpolationRectSize.x + cbFI.interpolationRectBase.x);
            }
            else
            {
                _4458 = _4446;
            }
            bool _4467;
            if (!_4458)
            {
                _4467 = _4221.y < cbFI.interpolationRectBase.y;
            }
            else
            {
                _4467 = _4458;
            }
            bool _4479;
            if (!_4467)
            {
                _4479 = _4221.y >= (cbFI.interpolationRectSize.y + cbFI.interpolationRectBase.y);
            }
            else
            {
                _4479 = _4467;
            }
            float4 _11243;
            if (_4479)
            {
                float4 _10860 = _4436;
                _10860.w = 0.0;
                _11243 = _10860;
            }
            else
            {
                _11243 = _4436;
            }
            int2 _4225 = int2(_2257 + uint2(1u));
            float4 _4532 = r_output.read(uint2(_4225), 0) * _2458;
            _4532.w = fast::clamp(1.0 - _4532.w, 0.0, 1.0);
            int _4539 = _4225.x;
            bool _4542 = _4539 < cbFI.interpolationRectBase.x;
            bool _4554;
            if (!_4542)
            {
                _4554 = _4539 >= (cbFI.interpolationRectSize.x + cbFI.interpolationRectBase.x);
            }
            else
            {
                _4554 = _4542;
            }
            bool _4563;
            if (!_4554)
            {
                _4563 = _4225.y < cbFI.interpolationRectBase.y;
            }
            else
            {
                _4563 = _4554;
            }
            bool _4575;
            if (!_4563)
            {
                _4575 = _4225.y >= (cbFI.interpolationRectSize.y + cbFI.interpolationRectBase.y);
            }
            else
            {
                _4575 = _4563;
            }
            float4 _11247;
            if (_4575)
            {
                float4 _10869 = _4532;
                _10869.w = 0.0;
                _11247 = _10869;
            }
            else
            {
                _11247 = _4532;
            }
            float4 _11248;
            do
            {
                float _4643 = ((_11235.w + _11239.w) + _11243.w) + _11247.w;
                if (_4643 == 0.0)
                {
                    _11248 = float4(0.0);
                    break;
                }
                _11248 = ((((_11235 * _11235.w) + (_11239 * _11239.w)) + (_11243 * _11243.w)) + (_11247 * _11247.w)) / float4(_4643);
                break;
            } while(false);
            rw_inpainting_pyramid0.write(_11248, uint2((_2145 + int2(_2181, _2217))));
            if (cbInpaintingPyramid.mips <= 1u)
            {
                break;
            }
            float4 _4784 = spvQuadSwap(_11146, 0u);
            float4 _4786 = spvQuadSwap(_11146, 1u);
            float4 _4788 = spvQuadSwap(_11146, 2u);
            float4 _11369;
            do
            {
                float _4804 = _4784.w;
                float _4806 = _4786.w;
                float _4808 = _4788.w;
                float _4820 = ((_11146.w + _4804) + _4806) + _4808;
                if (_4820 == 0.0)
                {
                    _11369 = float4(0.0);
                    break;
                }
                _11369 = ((((_11146 * _11146.w) + (_4784 * _4804)) + (_4786 * _4806)) + (_4788 * _4808)) / float4(_4820);
                break;
            } while(false);
            float4 _4861 = spvQuadSwap(_11180, 0u);
            float4 _4863 = spvQuadSwap(_11180, 1u);
            float4 _4865 = spvQuadSwap(_11180, 2u);
            float4 _11444;
            do
            {
                float _4881 = _4861.w;
                float _4883 = _4863.w;
                float _4885 = _4865.w;
                float _4897 = ((_11180.w + _4881) + _4883) + _4885;
                if (_4897 == 0.0)
                {
                    _11444 = float4(0.0);
                    break;
                }
                _11444 = ((((_11180 * _11180.w) + (_4861 * _4881)) + (_4863 * _4883)) + (_4865 * _4885)) / float4(_4897);
                break;
            } while(false);
            float4 _4938 = spvQuadSwap(_11214, 0u);
            float4 _4940 = spvQuadSwap(_11214, 1u);
            float4 _4942 = spvQuadSwap(_11214, 2u);
            float4 _11490;
            do
            {
                float _4958 = _4938.w;
                float _4960 = _4940.w;
                float _4962 = _4942.w;
                float _4974 = ((_11214.w + _4958) + _4960) + _4962;
                if (_4974 == 0.0)
                {
                    _11490 = float4(0.0);
                    break;
                }
                _11490 = ((((_11214 * _11214.w) + (_4938 * _4958)) + (_4940 * _4960)) + (_4942 * _4962)) / float4(_4974);
                break;
            } while(false);
            float4 _5015 = spvQuadSwap(_11248, 0u);
            float4 _5017 = spvQuadSwap(_11248, 1u);
            float4 _5019 = spvQuadSwap(_11248, 2u);
            float4 _11507;
            do
            {
                float _5035 = _5015.w;
                float _5037 = _5017.w;
                float _5039 = _5019.w;
                float _5051 = ((_11248.w + _5035) + _5037) + _5039;
                if (_5051 == 0.0)
                {
                    _11507 = float4(0.0);
                    break;
                }
                _11507 = ((((_11248 * _11248.w) + (_5015 * _5035)) + (_5017 * _5037)) + (_5019 * _5039)) / float4(_5051);
                break;
            } while(false);
            if ((gl_LocalInvocationIndex % 4u) == 0u)
            {
                int2 _2293 = int2(_1894 * uint2(16u));
                uint _2295 = _1944 / 2u;
                int _2296 = int(_2295);
                uint _2298 = _1950 / 2u;
                int _2299 = int(_2298);
                rw_inpainting_pyramid1.write(_11369, uint2((_2293 + int2(_2296, _2299))));
                spdIntermediateR[_2295][_2298] = _11369.x;
                spdIntermediateG[_2295][_2298] = _11369.y;
                spdIntermediateB[_2295][_2298] = _11369.z;
                spdIntermediateA[_2295][_2298] = _11369.w;
                uint _2319 = _2295 + 8u;
                int _2320 = int(_2319);
                rw_inpainting_pyramid1.write(_11444, uint2((_2293 + int2(_2320, _2299))));
                spdIntermediateR[_2319][_2298] = _11444.x;
                spdIntermediateG[_2319][_2298] = _11444.y;
                spdIntermediateB[_2319][_2298] = _11444.z;
                spdIntermediateA[_2319][_2298] = _11444.w;
                uint _2347 = _2298 + 8u;
                int _2348 = int(_2347);
                rw_inpainting_pyramid1.write(_11490, uint2((_2293 + int2(_2296, _2348))));
                spdIntermediateR[_2295][_2347] = _11490.x;
                spdIntermediateG[_2295][_2347] = _11490.y;
                spdIntermediateB[_2295][_2347] = _11490.z;
                spdIntermediateA[_2295][_2347] = _11490.w;
                rw_inpainting_pyramid1.write(_11507, uint2((_2293 + int2(_2320, _2348))));
                spdIntermediateR[_2319][_2347] = _11507.x;
                spdIntermediateG[_2319][_2347] = _11507.y;
                spdIntermediateB[_2319][_2347] = _11507.z;
                spdIntermediateA[_2319][_2347] = _11507.w;
            }
            break;
        } while(false);
        do
        {
            if (cbInpaintingPyramid.mips <= 2u)
            {
                break;
            }
            atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture, memory_order_seq_cst, thread_scope_threadgroup);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float4 _5711 = float4(spdIntermediateR[_1944][_1950], spdIntermediateG[_1944][_1950], spdIntermediateB[_1944][_1950], spdIntermediateA[_1944][_1950]);
            float4 _5724 = spvQuadSwap(_5711, 0u);
            float4 _5726 = spvQuadSwap(_5711, 1u);
            float4 _5728 = spvQuadSwap(_5711, 2u);
            float4 _11644;
            do
            {
                float _5744 = _5724.w;
                float _5746 = _5726.w;
                float _5748 = _5728.w;
                float _5760 = ((spdIntermediateA[_1944][_1950] + _5744) + _5746) + _5748;
                if (_5760 == 0.0)
                {
                    _11644 = float4(0.0);
                    break;
                }
                _11644 = ((((_5711 * spdIntermediateA[_1944][_1950]) + (_5724 * _5744)) + (_5726 * _5746)) + (_5728 * _5748)) / float4(_5760);
                break;
            } while(false);
            bool _5666 = (gl_LocalInvocationIndex % 4u) == 0u;
            if (_5666)
            {
                uint _5676 = _1950 / 2u;
                rw_inpainting_pyramid2.write(_11644, uint2((int2(_1894 * uint2(8u)) + int2(int(_1944 / 2u), int(_5676)))));
                uint _5688 = _1944 + (_5676 % 2u);
                spdIntermediateR[_5688][_1950] = _11644.x;
                spdIntermediateG[_5688][_1950] = _11644.y;
                spdIntermediateB[_5688][_1950] = _11644.z;
                spdIntermediateA[_5688][_1950] = _11644.w;
            }
            if (cbInpaintingPyramid.mips <= 3u)
            {
                break;
            }
            atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture, memory_order_seq_cst, thread_scope_threadgroup);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (gl_LocalInvocationIndex < 64u)
            {
                uint _5930 = _2135 + (_1950 % 2u);
                float4 _5985 = float4(spdIntermediateR[_5930][_2138], spdIntermediateG[_5930][_2138], spdIntermediateB[_5930][_2138], spdIntermediateA[_5930][_2138]);
                float4 _5998 = spvQuadSwap(_5985, 0u);
                float4 _6000 = spvQuadSwap(_5985, 1u);
                float4 _6002 = spvQuadSwap(_5985, 2u);
                float4 _11658;
                do
                {
                    float _6018 = _5998.w;
                    float _6020 = _6000.w;
                    float _6022 = _6002.w;
                    float _6034 = ((spdIntermediateA[_5930][_2138] + _6018) + _6020) + _6022;
                    if (_6034 == 0.0)
                    {
                        _11658 = float4(0.0);
                        break;
                    }
                    _11658 = ((((_5985 * spdIntermediateA[_5930][_2138]) + (_5998 * _6018)) + (_6000 * _6020)) + (_6002 * _6022)) / float4(_6034);
                    break;
                } while(false);
                if (_5666)
                {
                    uint _5948 = _1950 / 2u;
                    rw_inpainting_pyramid3.write(_11658, uint2((int2(_1894 * uint2(4u)) + int2(int(_1944 / 2u), int(_5948)))));
                    uint _5960 = _2135 + _5948;
                    spdIntermediateR[_5960][_2138] = _11658.x;
                    spdIntermediateG[_5960][_2138] = _11658.y;
                    spdIntermediateB[_5960][_2138] = _11658.z;
                    spdIntermediateA[_5960][_2138] = _11658.w;
                }
            }
            if (cbInpaintingPyramid.mips <= 4u)
            {
                break;
            }
            atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture, memory_order_seq_cst, thread_scope_threadgroup);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (gl_LocalInvocationIndex < 16u)
            {
                uint _6203 = (_1944 * 4u) + _1950;
                uint _6205 = _1950 * 4u;
                float4 _6255 = float4(spdIntermediateR[_6203][_6205], spdIntermediateG[_6203][_6205], spdIntermediateB[_6203][_6205], spdIntermediateA[_6203][_6205]);
                float4 _6268 = spvQuadSwap(_6255, 0u);
                float4 _6270 = spvQuadSwap(_6255, 1u);
                float4 _6272 = spvQuadSwap(_6255, 2u);
                float4 _11672;
                do
                {
                    float _6288 = _6268.w;
                    float _6290 = _6270.w;
                    float _6292 = _6272.w;
                    float _6304 = ((spdIntermediateA[_6203][_6205] + _6288) + _6290) + _6292;
                    if (_6304 == 0.0)
                    {
                        _11672 = float4(0.0);
                        break;
                    }
                    _11672 = ((((_6255 * spdIntermediateA[_6203][_6205]) + (_6268 * _6288)) + (_6270 * _6290)) + (_6272 * _6292)) / float4(_6304);
                    break;
                } while(false);
                if (_5666)
                {
                    uint _6218 = _1944 / 2u;
                    rw_inpainting_pyramid4.write(_11672, uint2((int2(_1894 * uint2(2u)) + int2(int(_6218), int(_1950 / 2u)))));
                    uint _6232 = _6218 + _1950;
                    spdIntermediateR[_6232][0u] = _11672.x;
                    spdIntermediateG[_6232][0u] = _11672.y;
                    spdIntermediateB[_6232][0u] = _11672.z;
                    spdIntermediateA[_6232][0u] = _11672.w;
                }
            }
            if (cbInpaintingPyramid.mips <= 5u)
            {
                break;
            }
            atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture, memory_order_seq_cst, thread_scope_threadgroup);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (gl_LocalInvocationIndex < 4u)
            {
                float4 _6501 = float4(spdIntermediateR[gl_LocalInvocationIndex][0u], spdIntermediateG[gl_LocalInvocationIndex][0u], spdIntermediateB[gl_LocalInvocationIndex][0u], spdIntermediateA[gl_LocalInvocationIndex][0u]);
                float4 _6514 = spvQuadSwap(_6501, 0u);
                float4 _6516 = spvQuadSwap(_6501, 1u);
                float4 _6518 = spvQuadSwap(_6501, 2u);
                float4 _11686;
                do
                {
                    float _6534 = _6514.w;
                    float _6536 = _6516.w;
                    float _6538 = _6518.w;
                    float _6550 = ((spdIntermediateA[gl_LocalInvocationIndex][0u] + _6534) + _6536) + _6538;
                    if (_6550 == 0.0)
                    {
                        _11686 = float4(0.0);
                        break;
                    }
                    _11686 = ((((_6501 * spdIntermediateA[gl_LocalInvocationIndex][0u]) + (_6514 * _6534)) + (_6516 * _6536)) + (_6518 * _6538)) / float4(_6550);
                    break;
                } while(false);
                if (_5666)
                {
                    rw_inpainting_pyramid5.write(_11686, uint2(int2(_1894)));
                }
            }
            break;
        } while(false);
        if (cbInpaintingPyramid.mips <= 6u)
        {
            break;
        }
        if (gl_LocalInvocationIndex == 0u)
        {
            uint _6701 = atomic_fetch_add_explicit((coherent device atomic_uint*)&rw_counters.data[0], 1u, memory_order_relaxed);
            spdCounter = _6701;
        }
        atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture, memory_order_seq_cst, thread_scope_threadgroup);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (spdCounter != (cbInpaintingPyramid.numWorkGroups - 1u))
        {
            break;
        }
        uint _6760;
        uint _6764;
        rw_counters.data[0] = 0u;
        do
        {
            _6760 = _1944 * 4u;
            int _6762 = int(_6760);
            _6764 = _1950 * 4u;
            int _6766 = int(_6764);
            int2 _6767 = int2(_6762, _6766);
            uint2 _6778 = uint2(_6767);
            spvImageFence(rw_inpainting_pyramid5);
            float4 _6992 = rw_inpainting_pyramid5.read(uint2(_6767));
            spvImageFence(rw_inpainting_pyramid5);
            float4 _7096 = rw_inpainting_pyramid5.read(uint2(int2(_6778 + uint2(0u, 1u))));
            spvImageFence(rw_inpainting_pyramid5);
            float4 _7200 = rw_inpainting_pyramid5.read(uint2(int2(_6778 + uint2(1u, 0u))));
            spvImageFence(rw_inpainting_pyramid5);
            float4 _7304 = rw_inpainting_pyramid5.read(uint2(int2(_6778 + uint2(1u))));
            float4 _11691;
            do
            {
                float _7365 = _6992.w;
                float _7367 = _7096.w;
                float _7369 = _7200.w;
                float _7371 = _7304.w;
                float _7383 = ((_7365 + _7367) + _7369) + _7371;
                if (_7383 == 0.0)
                {
                    _11691 = float4(0.0);
                    break;
                }
                _11691 = ((((_6992 * _7365) + (_7096 * _7367)) + (_7200 * _7369)) + (_7304 * _7371)) / float4(_7383);
                break;
            } while(false);
            rw_inpainting_pyramid6.write(_11691, uint2(_2140));
            int _6788 = int(_6760 + 2u);
            int2 _6793 = int2(_6788, _6766);
            int _6797 = int(_2135 + 1u);
            uint2 _6804 = uint2(_6793);
            spvImageFence(rw_inpainting_pyramid5);
            float4 _7619 = rw_inpainting_pyramid5.read(uint2(_6793));
            spvImageFence(rw_inpainting_pyramid5);
            float4 _7723 = rw_inpainting_pyramid5.read(uint2(int2(_6804 + uint2(0u, 1u))));
            spvImageFence(rw_inpainting_pyramid5);
            float4 _7827 = rw_inpainting_pyramid5.read(uint2(int2(_6804 + uint2(1u, 0u))));
            spvImageFence(rw_inpainting_pyramid5);
            float4 _7931 = rw_inpainting_pyramid5.read(uint2(int2(_6804 + uint2(1u))));
            float4 _11701;
            do
            {
                float _7992 = _7619.w;
                float _7994 = _7723.w;
                float _7996 = _7827.w;
                float _7998 = _7931.w;
                float _8010 = ((_7992 + _7994) + _7996) + _7998;
                if (_8010 == 0.0)
                {
                    _11701 = float4(0.0);
                    break;
                }
                _11701 = ((((_7619 * _7992) + (_7723 * _7994)) + (_7827 * _7996)) + (_7931 * _7998)) / float4(_8010);
                break;
            } while(false);
            rw_inpainting_pyramid6.write(_11701, uint2(int2(_6797, _2139)));
            int _6818 = int(_6764 + 2u);
            int2 _6819 = int2(_6762, _6818);
            int _6827 = int(_2138 + 1u);
            uint2 _6830 = uint2(_6819);
            spvImageFence(rw_inpainting_pyramid5);
            float4 _8246 = rw_inpainting_pyramid5.read(uint2(_6819));
            spvImageFence(rw_inpainting_pyramid5);
            float4 _8350 = rw_inpainting_pyramid5.read(uint2(int2(_6830 + uint2(0u, 1u))));
            spvImageFence(rw_inpainting_pyramid5);
            float4 _8454 = rw_inpainting_pyramid5.read(uint2(int2(_6830 + uint2(1u, 0u))));
            spvImageFence(rw_inpainting_pyramid5);
            float4 _8558 = rw_inpainting_pyramid5.read(uint2(int2(_6830 + uint2(1u))));
            float4 _11711;
            do
            {
                float _8619 = _8246.w;
                float _8621 = _8350.w;
                float _8623 = _8454.w;
                float _8625 = _8558.w;
                float _8637 = ((_8619 + _8621) + _8623) + _8625;
                if (_8637 == 0.0)
                {
                    _11711 = float4(0.0);
                    break;
                }
                _11711 = ((((_8246 * _8619) + (_8350 * _8621)) + (_8454 * _8623)) + (_8558 * _8625)) / float4(_8637);
                break;
            } while(false);
            rw_inpainting_pyramid6.write(_11711, uint2(int2(_2136, _6827)));
            int2 _6845 = int2(_6788, _6818);
            uint2 _6856 = uint2(_6845);
            spvImageFence(rw_inpainting_pyramid5);
            float4 _8873 = rw_inpainting_pyramid5.read(uint2(_6845));
            spvImageFence(rw_inpainting_pyramid5);
            float4 _8977 = rw_inpainting_pyramid5.read(uint2(int2(_6856 + uint2(0u, 1u))));
            spvImageFence(rw_inpainting_pyramid5);
            float4 _9081 = rw_inpainting_pyramid5.read(uint2(int2(_6856 + uint2(1u, 0u))));
            spvImageFence(rw_inpainting_pyramid5);
            float4 _9185 = rw_inpainting_pyramid5.read(uint2(int2(_6856 + uint2(1u))));
            float4 _11721;
            do
            {
                float _9246 = _8873.w;
                float _9248 = _8977.w;
                float _9250 = _9081.w;
                float _9252 = _9185.w;
                float _9264 = ((_9246 + _9248) + _9250) + _9252;
                if (_9264 == 0.0)
                {
                    _11721 = float4(0.0);
                    break;
                }
                _11721 = ((((_8873 * _9246) + (_8977 * _9248)) + (_9081 * _9250)) + (_9185 * _9252)) / float4(_9264);
                break;
            } while(false);
            rw_inpainting_pyramid6.write(_11721, uint2(int2(_6797, _6827)));
            if (cbInpaintingPyramid.mips <= 7u)
            {
                break;
            }
            float4 _11727;
            do
            {
                float _9419 = ((_11691.w + _11701.w) + _11711.w) + _11721.w;
                if (_9419 == 0.0)
                {
                    _11727 = float4(0.0);
                    break;
                }
                _11727 = ((((_11691 * _11691.w) + (_11701 * _11701.w)) + (_11711 * _11711.w)) + (_11721 * _11721.w)) / float4(_9419);
                break;
            } while(false);
            rw_inpainting_pyramid7.write(_11727, uint2(_2150));
            spdIntermediateR[_1944][_1950] = _11727.x;
            spdIntermediateG[_1944][_1950] = _11727.y;
            spdIntermediateB[_1944][_1950] = _11727.z;
            spdIntermediateA[_1944][_1950] = _11727.w;
            break;
        } while(false);
        do
        {
            if (cbInpaintingPyramid.mips <= 8u)
            {
                break;
            }
            atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture, memory_order_seq_cst, thread_scope_threadgroup);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float4 _9716 = float4(spdIntermediateR[_1944][_1950], spdIntermediateG[_1944][_1950], spdIntermediateB[_1944][_1950], spdIntermediateA[_1944][_1950]);
            float4 _9729 = spvQuadSwap(_9716, 0u);
            float4 _9731 = spvQuadSwap(_9716, 1u);
            float4 _9733 = spvQuadSwap(_9716, 2u);
            float4 _11728;
            do
            {
                float _9749 = _9729.w;
                float _9751 = _9731.w;
                float _9753 = _9733.w;
                float _9765 = ((spdIntermediateA[_1944][_1950] + _9749) + _9751) + _9753;
                if (_9765 == 0.0)
                {
                    _11728 = float4(0.0);
                    break;
                }
                _11728 = ((((_9716 * spdIntermediateA[_1944][_1950]) + (_9729 * _9749)) + (_9731 * _9751)) + (_9733 * _9753)) / float4(_9765);
                break;
            } while(false);
            bool _9671 = (gl_LocalInvocationIndex % 4u) == 0u;
            if (_9671)
            {
                uint _9681 = _1950 / 2u;
                rw_inpainting_pyramid8.write(_11728, uint2(int2(int(_1944 / 2u), int(_9681))));
                uint _9693 = _1944 + (_9681 % 2u);
                spdIntermediateR[_9693][_1950] = _11728.x;
                spdIntermediateG[_9693][_1950] = _11728.y;
                spdIntermediateB[_9693][_1950] = _11728.z;
                spdIntermediateA[_9693][_1950] = _11728.w;
            }
            if (cbInpaintingPyramid.mips <= 9u)
            {
                break;
            }
            atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture, memory_order_seq_cst, thread_scope_threadgroup);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (gl_LocalInvocationIndex < 64u)
            {
                uint _9935 = _2135 + (_1950 % 2u);
                float4 _9990 = float4(spdIntermediateR[_9935][_2138], spdIntermediateG[_9935][_2138], spdIntermediateB[_9935][_2138], spdIntermediateA[_9935][_2138]);
                float4 _10003 = spvQuadSwap(_9990, 0u);
                float4 _10005 = spvQuadSwap(_9990, 1u);
                float4 _10007 = spvQuadSwap(_9990, 2u);
                float4 _11742;
                do
                {
                    float _10023 = _10003.w;
                    float _10025 = _10005.w;
                    float _10027 = _10007.w;
                    float _10039 = ((spdIntermediateA[_9935][_2138] + _10023) + _10025) + _10027;
                    if (_10039 == 0.0)
                    {
                        _11742 = float4(0.0);
                        break;
                    }
                    _11742 = ((((_9990 * spdIntermediateA[_9935][_2138]) + (_10003 * _10023)) + (_10005 * _10025)) + (_10007 * _10027)) / float4(_10039);
                    break;
                } while(false);
                if (_9671)
                {
                    uint _9953 = _1950 / 2u;
                    rw_inpainting_pyramid9.write(_11742, uint2(int2(int(_1944 / 2u), int(_9953))));
                    uint _9965 = _2135 + _9953;
                    spdIntermediateR[_9965][_2138] = _11742.x;
                    spdIntermediateG[_9965][_2138] = _11742.y;
                    spdIntermediateB[_9965][_2138] = _11742.z;
                    spdIntermediateA[_9965][_2138] = _11742.w;
                }
            }
            if (cbInpaintingPyramid.mips <= 10u)
            {
                break;
            }
            atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture, memory_order_seq_cst, thread_scope_threadgroup);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (gl_LocalInvocationIndex < 16u)
            {
                uint _10208 = _6760 + _1950;
                float4 _10260 = float4(spdIntermediateR[_10208][_6764], spdIntermediateG[_10208][_6764], spdIntermediateB[_10208][_6764], spdIntermediateA[_10208][_6764]);
                float4 _10273 = spvQuadSwap(_10260, 0u);
                float4 _10275 = spvQuadSwap(_10260, 1u);
                float4 _10277 = spvQuadSwap(_10260, 2u);
                float4 _11756;
                do
                {
                    float _10293 = _10273.w;
                    float _10295 = _10275.w;
                    float _10297 = _10277.w;
                    float _10309 = ((spdIntermediateA[_10208][_6764] + _10293) + _10295) + _10297;
                    if (_10309 == 0.0)
                    {
                        _11756 = float4(0.0);
                        break;
                    }
                    _11756 = ((((_10260 * spdIntermediateA[_10208][_6764]) + (_10273 * _10293)) + (_10275 * _10295)) + (_10277 * _10297)) / float4(_10309);
                    break;
                } while(false);
                if (_9671)
                {
                    uint _10223 = _1944 / 2u;
                    rw_inpainting_pyramid10.write(_11756, uint2(int2(int(_10223), int(_1950 / 2u))));
                    uint _10237 = _10223 + _1950;
                    spdIntermediateR[_10237][0u] = _11756.x;
                    spdIntermediateG[_10237][0u] = _11756.y;
                    spdIntermediateB[_10237][0u] = _11756.z;
                    spdIntermediateA[_10237][0u] = _11756.w;
                }
            }
            if (cbInpaintingPyramid.mips <= 11u)
            {
                break;
            }
            atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture, memory_order_seq_cst, thread_scope_threadgroup);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (gl_LocalInvocationIndex < 4u)
            {
                float4 _10506 = float4(spdIntermediateR[gl_LocalInvocationIndex][0u], spdIntermediateG[gl_LocalInvocationIndex][0u], spdIntermediateB[gl_LocalInvocationIndex][0u], spdIntermediateA[gl_LocalInvocationIndex][0u]);
                float4 _10519 = spvQuadSwap(_10506, 0u);
                float4 _10521 = spvQuadSwap(_10506, 1u);
                float4 _10523 = spvQuadSwap(_10506, 2u);
                float4 _11770;
                do
                {
                    float _10539 = _10519.w;
                    float _10541 = _10521.w;
                    float _10543 = _10523.w;
                    float _10555 = ((spdIntermediateA[gl_LocalInvocationIndex][0u] + _10539) + _10541) + _10543;
                    if (_10555 == 0.0)
                    {
                        _11770 = float4(0.0);
                        break;
                    }
                    _11770 = ((((_10506 * spdIntermediateA[gl_LocalInvocationIndex][0u]) + (_10519 * _10539)) + (_10521 * _10541)) + (_10523 * _10543)) / float4(_10555);
                    break;
                } while(false);
                if (_9671)
                {
                    rw_inpainting_pyramid11.write(_11770, uint2(int2(0)));
                }
            }
            break;
        } while(false);
        break;
    } while(false);
}


}

// ===== AMD FSR disocclusion_mask =====
namespace ism_disocclusion_mask {
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wmissing-braces"


using namespace metal;

template<typename T, size_t Num>
struct spvUnsafeArray
{
    T elements[Num ? Num : 1];
    
    thread T& operator [] (size_t pos) thread
    {
        return elements[pos];
    }
    constexpr const thread T& operator [] (size_t pos) const thread
    {
        return elements[pos];
    }
    
    device T& operator [] (size_t pos) device
    {
        return elements[pos];
    }
    constexpr const device T& operator [] (size_t pos) const device
    {
        return elements[pos];
    }
    
    constexpr const constant T& operator [] (size_t pos) const constant
    {
        return elements[pos];
    }
    
    threadgroup T& operator [] (size_t pos) threadgroup
    {
        return elements[pos];
    }
    constexpr const threadgroup T& operator [] (size_t pos) const threadgroup
    {
        return elements[pos];
    }
};

struct cbFI_t
{
    int2 renderSize;
    int2 displaySize;
    float2 displaySizeRcp;
    float cameraNear;
    float cameraFar;
    int2 upscalerTargetSize;
    int Mode;
    int reset;
    float4 fDeviceToViewDepth;
    float deltaTime;
    int HUDLessAttachedFactor;
    int2 distortionFieldSize;
    float2 opticalFlowScale;
    int opticalFlowBlockSize;
    uint dispatchFlags;
    int2 maxRenderSize;
    int opticalFlowHalfResMode;
    int NumInstances;
    int2 interpolationRectBase;
    int2 interpolationRectSize;
    packed_float3 debugBarColor;
    uint backBufferTransferFunction;
    float2 minMaxLuminance;
    float fTanHalfFOV;
    int _pad1;
    float2 fJitter;
    float2 fMotionVectorScale;
};

constant uint3 gl_WorkGroupSize [[maybe_unused]] = uint3(8u, 8u, 1u);

constant spvUnsafeArray<int2, 4> _2892 = spvUnsafeArray<int2, 4>({ int2(0), int2(1, 0), int2(0, 1), int2(1) });

[[host_name("ism_fsr_disocclusion_mask")]] kernel void ism_fsr_disocclusion_mask(constant cbFI_t& cbFI [[buffer(0)]], texture2d<float> r_dilated_depth [[texture(0)]], texture2d<uint> r_reconstructed_depth_previous_frame [[texture(1)]], texture2d<uint> r_reconstructed_depth_interpolated_frame [[texture(2)]], texture2d<uint> r_game_motion_vector_field_x [[texture(3)]], texture2d<uint> r_game_motion_vector_field_y [[texture(4)]], texture2d<float> r_inpainting_pyramid [[texture(5)]], texture2d<float> r_input_distortion_field [[texture(6)]], texture2d<float, access::write> rw_disocclusion_mask [[texture(7)]], sampler s_LinearClamp [[sampler(0)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    int2 _978 = int2(gl_GlobalInvocationID.xy);
    float _1112 = as_type<float>(r_reconstructed_depth_interpolated_frame.read(uint2(_978), 0).x);
    float2 _1031 = float2(_978) + float2(0.5);
    float2 _1033 = float2(cbFI.renderSize);
    float2 _1034 = _1031 / _1033;
    int2 _1148 = int2(_1031);
    uint4 _1220 = r_game_motion_vector_field_x.read(uint2(_1148), 0);
    uint _1221 = _1220.x;
    float _1238 = float((_1221 >> 21u) & 1023u) * 0.000977517105638980865478515625;
    float2 _2814;
    if ((isunordered(_1238, 0.0) || _1238 <= 0.0))
    {
        float4 _2812;
        int2 _2868;
        _2868 = cbFI.renderSize;
        _2812 = float4(0.0);
        int2 _1172;
        float4 _1371;
        int _2811 = 0;
        for (;;)
        {
            bool _1162 = _2811 < 11;
            bool _1168;
            if (_1162)
            {
                _1168 = _2812.w == 0.0;
            }
            else
            {
                _1168 = _1162;
            }
            if (_1168)
            {
                _1172 = _2868 / int2(2);
                float2 _1383 = (_1034 * float2(_1172)) - float2(0.5);
                int2 _1386 = int2(floor(_1383));
                float2 _1426 = fract(_1383);
                float _1395 = _1426.x;
                float _1396 = 1.0 - _1395;
                float _1398 = _1426.y;
                float _1399 = 1.0 - _1398;
                spvUnsafeArray<float, 4> _2711 = spvUnsafeArray<float, 4>({ _1396 * _1399, _1395 * _1399, _1396 * _1398, _1395 * _1398 });
                spvUnsafeArray<float, 4> _2599 = _2711;
                float _2871;
                float4 _2872;
                _2872 = float4(0.0);
                _2871 = 0.0;
                float _2889;
                float4 _2890;
                for (int _2870 = 0; _2870 < 4; _2872 = _2890, _2871 = _2889, _2870++)
                {
                    int2 _1332 = _1386 + _2892[_2870];
                    if (all(uint2(_1332) < uint2(_1172)))
                    {
                        float4 _1441 = r_inpainting_pyramid.read(uint2(_1332), _2811);
                        float _1340 = _1441.z;
                        float _1351 = (_2599[_2870] * float(_1340 > 0.0)) * _1340;
                        _2890 = _2872 + (_1441 * _1351);
                        _2889 = _2871 + _1351;
                    }
                    else
                    {
                        _2890 = _2872;
                        _2889 = _2871;
                    }
                }
                _1371 = _2872 / float4((_2871 > 0.0) ? _2871 : 1.0);
                _2868 = _1172;
                _2812 = _1371;
                _2811++;
                continue;
            }
            else
            {
                break;
            }
        }
        _2814 = _2812.xy;
    }
    else
    {
        _2814 = float2(float2(as_type<half2>(_1221)).x, float2(as_type<half2>(r_game_motion_vector_field_y.read(uint2(_1148), 0).x)).x);
    }
    float _1650 = cbFI.fDeviceToViewDepth.y / (_1112 - cbFI.fDeviceToViewDepth.x);
    float2 _1669 = ((_1034 + _2814) * _1033) - float2(0.5);
    int2 _1672 = int2(floor(_1669));
    float2 _1712 = fract(_1669);
    float _1681 = _1712.x;
    float _1682 = 1.0 - _1681;
    float _1684 = _1712.y;
    float _1685 = 1.0 - _1684;
    spvUnsafeArray<float, 4> _2693 = spvUnsafeArray<float, 4>({ _1682 * _1685, _1681 * _1685, _1682 * _1684, _1681 * _1684 });
    spvUnsafeArray<float, 4> _2576 = _2693;
    float _2846;
    float _2847;
    _2847 = 0.0;
    _2846 = 0.0;
    float _2875;
    float _2878;
    for (int _2845 = 0; _2845 < 4; _2847 = _2878, _2846 = _2875, _2845++)
    {
        int2 _1551 = _1672 + _2892[_2845];
        if (all(uint2(_1551) < uint2(cbFI.renderSize)))
        {
            float _2876;
            float _2879;
            if (_2576[_2845] > 0.001000000047497451305389404296875)
            {
                float _1796 = as_type<float>(r_reconstructed_depth_previous_frame.read(uint2((_1551 + int2(r_input_distortion_field.sample(s_LinearClamp, ((float2(_1551) + float2(0.5)) / _1033), level(0.0)).xy * _1033))), 0).x);
                float _1813 = cbFI.fDeviceToViewDepth.y / (_1796 - cbFI.fDeviceToViewDepth.x);
                float _1569 = _1650 - _1813;
                float _2877;
                float _2880;
                if (_1569 > 0.0)
                {
                    float _1880 = cbFI.fDeviceToViewDepth.y / (fast::max(_1796, _1112) - cbFI.fDeviceToViewDepth.x);
                    float2 _1892 = ((float2(int2(_1033 * 0.5)) / _1033) * float2(2.0, -2.0)) + float2(-1.0, 1.0);
                    _2880 = _2847 + (float(((((1.3699999726668465882539749145508e-05 * (length(float3((cbFI.fDeviceToViewDepth.z * (-1.0)) * _1880, cbFI.fDeviceToViewDepth.w * _1880, _1880)) / length(float3((cbFI.fDeviceToViewDepth.z * _1892.x) * _1880, (cbFI.fDeviceToViewDepth.w * _1892.y) * _1880, _1880)))) * length(_1033)) * fast::min(_1650, _1813)) / _1569) >= 1.0) * _2576[_2845]);
                    _2877 = _2846 + _2576[_2845];
                }
                else
                {
                    _2880 = _2847;
                    _2877 = _2846;
                }
                _2879 = _2880;
                _2876 = _2877;
            }
            else
            {
                _2879 = _2847;
                _2876 = _2846;
            }
            _2878 = _2879;
            _2875 = _2876;
        }
        else
        {
            _2878 = _2847;
            _2875 = _2846;
        }
    }
    float _2848;
    if (_2846 > 0.0)
    {
        _2848 = fast::clamp(1.0 - (_2847 / _2846), 0.0, 1.0);
    }
    else
    {
        _2848 = 0.0;
    }
    float2 _2167 = ((_1034 - _2814) * _1033) - float2(0.5);
    int2 _2170 = int2(floor(_2167));
    float2 _2210 = fract(_2167);
    float _2179 = _2210.x;
    float _2180 = 1.0 - _2179;
    float _2182 = _2210.y;
    float _2183 = 1.0 - _2182;
    spvUnsafeArray<float, 4> _2675 = spvUnsafeArray<float, 4>({ _2180 * _2183, _2179 * _2183, _2180 * _2182, _2179 * _2182 });
    spvUnsafeArray<float, 4> _2553 = _2675;
    float _2850;
    float _2851;
    _2851 = 0.0;
    _2850 = 0.0;
    float _2881;
    float _2884;
    for (int _2849 = 0; _2849 < 4; _2851 = _2884, _2850 = _2881, _2849++)
    {
        int2 _2049 = _2170 + _2892[_2849];
        if (all(uint2(_2049) < uint2(cbFI.renderSize)))
        {
            float _2882;
            float _2885;
            if (_2553[_2849] > 0.001000000047497451305389404296875)
            {
                float4 _2299 = r_dilated_depth.read(uint2((_2049 + int2(r_input_distortion_field.sample(s_LinearClamp, ((float2(_2049) + float2(0.5)) / _1033), level(0.0)).xy * _1033))), 0);
                float _2300 = _2299.x;
                float _2311 = cbFI.fDeviceToViewDepth.y / (_2300 - cbFI.fDeviceToViewDepth.x);
                float _2067 = _1650 - _2311;
                float _2883;
                float _2886;
                if (_2067 > 0.0)
                {
                    float _2378 = cbFI.fDeviceToViewDepth.y / (fast::max(_2300, _1112) - cbFI.fDeviceToViewDepth.x);
                    float2 _2390 = ((float2(int2(_1033 * 0.5)) / _1033) * float2(2.0, -2.0)) + float2(-1.0, 1.0);
                    _2886 = _2851 + (float(((((1.3699999726668465882539749145508e-05 * (length(float3((cbFI.fDeviceToViewDepth.z * (-1.0)) * _2378, cbFI.fDeviceToViewDepth.w * _2378, _2378)) / length(float3((cbFI.fDeviceToViewDepth.z * _2390.x) * _2378, (cbFI.fDeviceToViewDepth.w * _2390.y) * _2378, _2378)))) * length(_1033)) * fast::min(_1650, _2311)) / _2067) >= 1.0) * _2553[_2849]);
                    _2883 = _2850 + _2553[_2849];
                }
                else
                {
                    _2886 = _2851;
                    _2883 = _2850;
                }
                _2885 = _2886;
                _2882 = _2883;
            }
            else
            {
                _2885 = _2851;
                _2882 = _2850;
            }
            _2884 = _2885;
            _2881 = _2882;
        }
        else
        {
            _2884 = _2851;
            _2881 = _2850;
        }
    }
    float _2852;
    if (_2850 > 0.0)
    {
        _2852 = fast::clamp(1.0 - (_2851 / _2850), 0.0, 1.0);
    }
    else
    {
        _2852 = 0.0;
    }
    float2 _1060 = float2(float2(1.0 - _2848, 1.0 - _2852) >= float2(0.001000000047497451305389404296875));
    float2 _1063 = _2814 * 2.0;
    uint2 _2501 = uint2(cbFI.renderSize);
    rw_disocclusion_mask.write(float4(fast::clamp(_1060.x + float(!all(uint2(int2((_1034 + _1063) * _1033)) < _2501)), 0.0, 1.0), fast::clamp(_1060.y + float(!all(uint2(int2((_1034 - _1063) * _1033)) < _2501)), 0.0, 1.0), 0.0, 0.0), uint2(_978));
}


}

// ===== AMD FSR game_motion_vector_field =====
namespace ism_game_motion_vector_field {
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wmissing-braces"


using namespace metal;

template<typename T, size_t Num>
struct spvUnsafeArray
{
    T elements[Num ? Num : 1];
    
    thread T& operator [] (size_t pos) thread
    {
        return elements[pos];
    }
    constexpr const thread T& operator [] (size_t pos) const thread
    {
        return elements[pos];
    }
    
    device T& operator [] (size_t pos) device
    {
        return elements[pos];
    }
    constexpr const device T& operator [] (size_t pos) const device
    {
        return elements[pos];
    }
    
    constexpr const constant T& operator [] (size_t pos) const constant
    {
        return elements[pos];
    }
    
    threadgroup T& operator [] (size_t pos) threadgroup
    {
        return elements[pos];
    }
    constexpr const threadgroup T& operator [] (size_t pos) const threadgroup
    {
        return elements[pos];
    }
};

struct cbFI_t
{
    int2 renderSize;
    int2 displaySize;
    float2 displaySizeRcp;
    float cameraNear;
    float cameraFar;
    int2 upscalerTargetSize;
    int Mode;
    int reset;
    float4 fDeviceToViewDepth;
    float deltaTime;
    int HUDLessAttachedFactor;
    int2 distortionFieldSize;
    float2 opticalFlowScale;
    int opticalFlowBlockSize;
    uint dispatchFlags;
    int2 maxRenderSize;
    int opticalFlowHalfResMode;
    int NumInstances;
    int2 interpolationRectBase;
    int2 interpolationRectSize;
    packed_float3 debugBarColor;
    uint backBufferTransferFunction;
    float2 minMaxLuminance;
    float fTanHalfFOV;
    int _pad1;
    float2 fJitter;
    float2 fMotionVectorScale;
};

constant uint3 gl_WorkGroupSize [[maybe_unused]] = uint3(8u, 8u, 1u);
constant float3 _3213 = {};

constant spvUnsafeArray<int2, 4> _3262 = spvUnsafeArray<int2, 4>({ int2(0), int2(1, 0), int2(0, 1), int2(1) });

[[host_name("ism_fsr_game_motion_vector_field")]] kernel void ism_fsr_game_motion_vector_field(constant cbFI_t& cbFI [[buffer(0)]], texture2d<float> r_previous_interpolation_source [[texture(0)]], texture2d<float> r_current_interpolation_source [[texture(1)]], texture2d<float> r_dilated_motion_vectors [[texture(2)]], texture2d<float> r_dilated_depth [[texture(3)]], texture2d<float> r_input_distortion_field [[texture(4)]], texture2d<uint, access::read_write> rw_game_motion_vector_field_x [[texture(5)]], texture2d<uint, access::read_write> rw_game_motion_vector_field_y [[texture(6)]], sampler s_LinearClamp [[sampler(0)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    int2 _1039 = int2(gl_GlobalInvocationID.xy);
    float2 _1136 = float2(cbFI.renderSize);
    float2 _1137 = (float2(_1039) + float2(0.5)) / _1136;
    float2 _1148 = float2(cbFI.displaySize);
    float2 _1154 = float2(cbFI.interpolationRectSize) / _1148;
    float2 _1159 = (float2(cbFI.interpolationRectBase) / _1148) + (_1137 * _1154);
    int2 _1162 = _1039 + int2(r_input_distortion_field.sample(s_LinearClamp, _1137, level(0.0)).xy * _1136);
    float2 _1375 = r_dilated_motion_vectors.read(uint2(_1162), 0).xy;
    float2 _1169 = _1375 * 0.5;
    float2 _1172 = _1137 + _1169;
    float _1414 = powr(cbFI.fDeviceToViewDepth.y / (r_dilated_depth.read(uint2(_1162), 0).x - cbFI.fDeviceToViewDepth.x), 0.3300000131130218505859375);
    uint _1419 = max(1u, uint((1.0 - (_1414 * (1.0 / (1.0 + _1414)))) * 1023.0));
    float2 _1181 = _1159 + (_1375 * _1154);
    float3 _1427 = r_previous_interpolation_source.sample(s_LinearClamp, _1181, level(0.0)).xyz;
    float3 _1435 = r_current_interpolation_source.sample(s_LinearClamp, _1159, level(0.0)).xyz;
    float _3216;
    switch (cbFI.backBufferTransferFunction)
    {
        case 0u:
        {
            float3 _3215;
            switch (cbFI.backBufferTransferFunction)
            {
                case 0u:
                {
                    float3 _1546 = fast::clamp((_1427 - float3(0.040449999272823333740234375)) * float3(as_type<float>(0xff800000u /* -inf */)), float3(0.0), float3(1.0));
                    float3 _1519 = powr((_1427 * float3(0.94786727428436279296875)) + float3(0.0521326996386051177978515625), float3(2.400000095367431640625));
                    _3215 = (_1546 * (_1427 * float3(0.077399380505084991455078125))) + (((-_1546) * _1519) + _1519);
                    break;
                }
                case 1u:
                {
                    float3 _1580 = powr(_1427, float3(0.01268330030143260955810546875));
                    _3215 = powr(fast::clamp(_1580 - float3(0.835937976837158203125), float3(0.0), float3(1.0)) / (float3(18.85160064697265625) - (float3(18.6875) * _1580)), float3(6.277390003204345703125)) * (10000.0 / cbFI.minMaxLuminance.y);
                    break;
                }
                case 2u:
                {
                    _3215 = (_1427 - float3(cbFI.minMaxLuminance.x * 0.012500000186264514923095703125)) / float3((cbFI.minMaxLuminance.y - cbFI.minMaxLuminance.x) * 0.012500000186264514923095703125);
                    break;
                }
                default:
                {
                    _3215 = _3213;
                    break;
                }
            }
            _3216 = dot(_3215, float3(0.2125999927520751953125, 0.715200006961822509765625, 0.072200000286102294921875));
            break;
        }
        case 1u:
        {
            float3 _3214;
            switch (cbFI.backBufferTransferFunction)
            {
                case 0u:
                {
                    float3 _1752 = fast::clamp((_1427 - float3(0.040449999272823333740234375)) * float3(as_type<float>(0xff800000u /* -inf */)), float3(0.0), float3(1.0));
                    float3 _1725 = powr((_1427 * float3(0.94786727428436279296875)) + float3(0.0521326996386051177978515625), float3(2.400000095367431640625));
                    _3214 = (_1752 * (_1427 * float3(0.077399380505084991455078125))) + (((-_1752) * _1725) + _1725);
                    break;
                }
                case 1u:
                {
                    float3 _1786 = powr(_1427, float3(0.01268330030143260955810546875));
                    _3214 = powr(fast::clamp(_1786 - float3(0.835937976837158203125), float3(0.0), float3(1.0)) / (float3(18.85160064697265625) - (float3(18.6875) * _1786)), float3(6.277390003204345703125)) * (10000.0 / cbFI.minMaxLuminance.y);
                    break;
                }
                case 2u:
                {
                    _3214 = (_1427 - float3(cbFI.minMaxLuminance.x * 0.012500000186264514923095703125)) / float3((cbFI.minMaxLuminance.y - cbFI.minMaxLuminance.x) * 0.012500000186264514923095703125);
                    break;
                }
                default:
                {
                    _3214 = _3213;
                    break;
                }
            }
            _3216 = ((0.2626999914646148681640625 * _3214.x) + (0.677999973297119140625 * _3214.y)) + (0.0593000017106533050537109375 * _3214.z);
            break;
        }
        case 2u:
        {
            float3 _3212;
            switch (cbFI.backBufferTransferFunction)
            {
                case 0u:
                {
                    float3 _1969 = fast::clamp((_1427 - float3(0.040449999272823333740234375)) * float3(as_type<float>(0xff800000u /* -inf */)), float3(0.0), float3(1.0));
                    float3 _1942 = powr((_1427 * float3(0.94786727428436279296875)) + float3(0.0521326996386051177978515625), float3(2.400000095367431640625));
                    _3212 = (_1969 * (_1427 * float3(0.077399380505084991455078125))) + (((-_1969) * _1942) + _1942);
                    break;
                }
                case 1u:
                {
                    float3 _2003 = powr(_1427, float3(0.01268330030143260955810546875));
                    _3212 = powr(fast::clamp(_2003 - float3(0.835937976837158203125), float3(0.0), float3(1.0)) / (float3(18.85160064697265625) - (float3(18.6875) * _2003)), float3(6.277390003204345703125)) * (10000.0 / cbFI.minMaxLuminance.y);
                    break;
                }
                case 2u:
                {
                    _3212 = (_1427 - float3(cbFI.minMaxLuminance.x * 0.012500000186264514923095703125)) / float3((cbFI.minMaxLuminance.y - cbFI.minMaxLuminance.x) * 0.012500000186264514923095703125);
                    break;
                }
                default:
                {
                    _3212 = _3213;
                    break;
                }
            }
            _3216 = dot(_3212, float3(0.2125999927520751953125, 0.715200006961822509765625, 0.072200000286102294921875));
            break;
        }
        default:
        {
            _3216 = 0.0;
            break;
        }
    }
    float _1187 = 0.001000000047497451305389404296875 + _3216;
    float _3232;
    switch (cbFI.backBufferTransferFunction)
    {
        case 0u:
        {
            float3 _3227;
            switch (cbFI.backBufferTransferFunction)
            {
                case 0u:
                {
                    float3 _2203 = fast::clamp((_1435 - float3(0.040449999272823333740234375)) * float3(as_type<float>(0xff800000u /* -inf */)), float3(0.0), float3(1.0));
                    float3 _2176 = powr((_1435 * float3(0.94786727428436279296875)) + float3(0.0521326996386051177978515625), float3(2.400000095367431640625));
                    _3227 = (_2203 * (_1435 * float3(0.077399380505084991455078125))) + (((-_2203) * _2176) + _2176);
                    break;
                }
                case 1u:
                {
                    float3 _2237 = powr(_1435, float3(0.01268330030143260955810546875));
                    _3227 = powr(fast::clamp(_2237 - float3(0.835937976837158203125), float3(0.0), float3(1.0)) / (float3(18.85160064697265625) - (float3(18.6875) * _2237)), float3(6.277390003204345703125)) * (10000.0 / cbFI.minMaxLuminance.y);
                    break;
                }
                case 2u:
                {
                    _3227 = (_1435 - float3(cbFI.minMaxLuminance.x * 0.012500000186264514923095703125)) / float3((cbFI.minMaxLuminance.y - cbFI.minMaxLuminance.x) * 0.012500000186264514923095703125);
                    break;
                }
                default:
                {
                    _3227 = _3213;
                    break;
                }
            }
            _3232 = dot(_3227, float3(0.2125999927520751953125, 0.715200006961822509765625, 0.072200000286102294921875));
            break;
        }
        case 1u:
        {
            float3 _3222;
            switch (cbFI.backBufferTransferFunction)
            {
                case 0u:
                {
                    float3 _2409 = fast::clamp((_1435 - float3(0.040449999272823333740234375)) * float3(as_type<float>(0xff800000u /* -inf */)), float3(0.0), float3(1.0));
                    float3 _2382 = powr((_1435 * float3(0.94786727428436279296875)) + float3(0.0521326996386051177978515625), float3(2.400000095367431640625));
                    _3222 = (_2409 * (_1435 * float3(0.077399380505084991455078125))) + (((-_2409) * _2382) + _2382);
                    break;
                }
                case 1u:
                {
                    float3 _2443 = powr(_1435, float3(0.01268330030143260955810546875));
                    _3222 = powr(fast::clamp(_2443 - float3(0.835937976837158203125), float3(0.0), float3(1.0)) / (float3(18.85160064697265625) - (float3(18.6875) * _2443)), float3(6.277390003204345703125)) * (10000.0 / cbFI.minMaxLuminance.y);
                    break;
                }
                case 2u:
                {
                    _3222 = (_1435 - float3(cbFI.minMaxLuminance.x * 0.012500000186264514923095703125)) / float3((cbFI.minMaxLuminance.y - cbFI.minMaxLuminance.x) * 0.012500000186264514923095703125);
                    break;
                }
                default:
                {
                    _3222 = _3213;
                    break;
                }
            }
            _3232 = ((0.2626999914646148681640625 * _3222.x) + (0.677999973297119140625 * _3222.y)) + (0.0593000017106533050537109375 * _3222.z);
            break;
        }
        case 2u:
        {
            float3 _3217;
            switch (cbFI.backBufferTransferFunction)
            {
                case 0u:
                {
                    float3 _2626 = fast::clamp((_1435 - float3(0.040449999272823333740234375)) * float3(as_type<float>(0xff800000u /* -inf */)), float3(0.0), float3(1.0));
                    float3 _2599 = powr((_1435 * float3(0.94786727428436279296875)) + float3(0.0521326996386051177978515625), float3(2.400000095367431640625));
                    _3217 = (_2626 * (_1435 * float3(0.077399380505084991455078125))) + (((-_2626) * _2599) + _2599);
                    break;
                }
                case 1u:
                {
                    float3 _2660 = powr(_1435, float3(0.01268330030143260955810546875));
                    _3217 = powr(fast::clamp(_2660 - float3(0.835937976837158203125), float3(0.0), float3(1.0)) / (float3(18.85160064697265625) - (float3(18.6875) * _2660)), float3(6.277390003204345703125)) * (10000.0 / cbFI.minMaxLuminance.y);
                    break;
                }
                case 2u:
                {
                    _3217 = (_1435 - float3(cbFI.minMaxLuminance.x * 0.012500000186264514923095703125)) / float3((cbFI.minMaxLuminance.y - cbFI.minMaxLuminance.x) * 0.012500000186264514923095703125);
                    break;
                }
                default:
                {
                    _3217 = _3213;
                    break;
                }
            }
            _3232 = dot(_3217, float3(0.2125999927520751953125, 0.715200006961822509765625, 0.072200000286102294921875));
            break;
        }
        default:
        {
            _3232 = 0.0;
            break;
        }
    }
    float _1190 = 0.001000000047497451305389404296875 + _3232;
    float _2772 = fast::max(_1187, _1190);
    float _3233;
    if (_2772 != 0.0)
    {
        _3233 = fast::min(_1187, _1190) / _2772;
    }
    else
    {
        _3233 = 0.0;
    }
    float _2790 = _1181.x;
    bool _2791 = _2790 > 0.0;
    bool _2797;
    if (_2791)
    {
        _2797 = _2790 < 1.0;
    }
    else
    {
        _2797 = _2791;
    }
    bool _2809;
    if (_2797)
    {
        float _2800 = _1181.y;
        bool _2801 = _2800 > 0.0;
        bool _2807;
        if (_2801)
        {
            _2807 = _2800 < 1.0;
        }
        else
        {
            _2807 = _2801;
        }
        _2809 = _2807;
    }
    else
    {
        _2809 = _2797;
    }
    uint _2827 = (2147483648u | ((_1419 & 1023u) << 21u)) | (((uint(rint(powr(_3233, 1.0) * 31.0)) * uint(_2809)) & 31u) << 16u);
    uint _2845 = as_type<uint>(half2(float2(_1169.x, 0.0)));
    uint _2832 = _2827 | _2845;
    uint _2850 = as_type<uint>(half2(float2(_1169.y, 0.0)));
    uint _2837 = _2827 | _2850;
    int2 _2868 = int2(floor((_1172 * _1136) - float2(0.5)));
    for (int _3234 = 0; _3234 < 4; _3234++)
    {
        int2 _1224 = _2868 + _3262[_3234];
        if (all(uint2(_1224) < uint2(cbFI.renderSize)))
        {
            uint _2926 = rw_game_motion_vector_field_x.atomic_fetch_max(uint2(_1224), _2832).x;
            uint _2931 = rw_game_motion_vector_field_y.atomic_fetch_max(uint2(_1224), _2837).x;
        }
    }
    if (length(_1169 * _1136) > 0.001000000047497451305389404296875)
    {
        float _1249 = length(float2(1.0) / _1136);
        float2 _1251 = fast::normalize(_1375);
        float _2944 = fast::min(length(_1169), 0.707106769084930419921875);
        bool _3239;
        uint _3241;
        float _3235 = _1249;
        bool _3236 = true;
        uint _3242 = 0u;
        for (; (_3235 <= _2944) && _3236; _3236 = _3239, _3235 += _1249, _3242 = _3241)
        {
            float2 _1268 = _1172 - (_1251 * _3235);
            int2 _2962 = int2(floor((_1268 * _1136) - float2(0.5)));
            uint _3030 = ((max(1u, (1023u - _1419)) & 1023u) << 21u) | ((uint(fast::max(0.0, dot(fast::normalize(float2(0.5) - _1268), _1251)) * 31.0) & 31u) << 16u);
            uint _3035 = _3030 | _2845;
            uint _3040 = _3030 | _2850;
            _3241 = _3242;
            _3239 = _3236;
            bool _3248;
            uint _3249;
            for (int _3237 = 0; _3237 < 1; _3241 = _3249, _3239 = _3248, _3237++)
            {
                int2 _1299 = _2962 + _3262[_3237];
                bool _1306;
                if (_3239)
                {
                    _1306 = all(uint2(_1299) < uint2(cbFI.renderSize));
                }
                else
                {
                    _1306 = _3239;
                }
                if (_1306)
                {
                    uint _3077 = rw_game_motion_vector_field_x.atomic_fetch_max(uint2(_1299), _3035).x;
                    uint _3082 = rw_game_motion_vector_field_y.atomic_fetch_max(uint2(_1299), _3040).x;
                    uint _1316 = _3241 + uint((max(_3077, _3082) & 2147483648u) != 0u);
                    _3249 = _1316;
                    _3248 = _1306 && (_1316 <= 3u);
                }
                else
                {
                    _3249 = _3241;
                    _3248 = _1306;
                }
            }
        }
    }
}


}

// ===== AMD FSR inpainting =====
namespace ism_inpainting {
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wmissing-braces"


using namespace metal;

template<typename T, size_t Num>
struct spvUnsafeArray
{
    T elements[Num ? Num : 1];
    
    thread T& operator [] (size_t pos) thread
    {
        return elements[pos];
    }
    constexpr const thread T& operator [] (size_t pos) const thread
    {
        return elements[pos];
    }
    
    device T& operator [] (size_t pos) device
    {
        return elements[pos];
    }
    constexpr const device T& operator [] (size_t pos) const device
    {
        return elements[pos];
    }
    
    constexpr const constant T& operator [] (size_t pos) const constant
    {
        return elements[pos];
    }
    
    threadgroup T& operator [] (size_t pos) threadgroup
    {
        return elements[pos];
    }
    constexpr const threadgroup T& operator [] (size_t pos) const threadgroup
    {
        return elements[pos];
    }
};

template <typename ImageT>
void spvImageFence(ImageT img) { img.fence(); }

struct cbFI_t
{
    int2 renderSize;
    int2 displaySize;
    float2 displaySizeRcp;
    float cameraNear;
    float cameraFar;
    int2 upscalerTargetSize;
    int Mode;
    int reset;
    float4 fDeviceToViewDepth;
    float deltaTime;
    int HUDLessAttachedFactor;
    int2 distortionFieldSize;
    float2 opticalFlowScale;
    int opticalFlowBlockSize;
    uint dispatchFlags;
    int2 maxRenderSize;
    int opticalFlowHalfResMode;
    int NumInstances;
    int2 interpolationRectBase;
    int2 interpolationRectSize;
    packed_float3 debugBarColor;
    uint backBufferTransferFunction;
    float2 minMaxLuminance;
    float fTanHalfFOV;
    int _pad1;
    float2 fJitter;
    float2 fMotionVectorScale;
};

constant uint3 gl_WorkGroupSize [[maybe_unused]] = uint3(8u, 8u, 1u);
constant float3 _2074 = {};
constant float4 _2165 = {};

constant spvUnsafeArray<int2, 4> _2150 = spvUnsafeArray<int2, 4>({ int2(0), int2(1, 0), int2(0, 1), int2(1) });

[[host_name("ism_fsr_inpainting")]] kernel void ism_fsr_inpainting(constant cbFI_t& cbFI [[buffer(0)]], texture2d<float> r_current_interpolation_source [[texture(0)]], texture2d<uint> r_optical_flow_scd [[texture(1)]], texture2d<float> r_inpainting_pyramid [[texture(2)]], texture2d<float> r_present_backbuffer [[texture(3)]], texture2d<float, access::read_write> rw_output [[texture(4)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    int2 _922 = int2(gl_GlobalInvocationID.xy);
    spvImageFence(rw_output);
    float4 _1085 = rw_output.read(uint2(_922));
    float _975 = _1085.w;
    bool _977 = _975 > 0.001000000047497451305389404296875;
    float4 _2088;
    if (_977)
    {
        float2 _1105 = (float2(_922) + float2(0.5)) / float2(cbFI.displaySize);
        float4 _2070;
        int2 _2126;
        _2126 = cbFI.displaySize;
        _2070 = float4(0.0);
        int2 _1114;
        float4 _2133;
        for (int _2069 = 0; _2069 < 10; _2126 = _1114, _2070 = _2133, _2069++)
        {
            _1114 = _2126 / int2(2);
            float2 _1235 = (_1105 * float2(_1114)) - float2(0.5);
            int2 _1238 = int2(floor(_1235));
            float2 _1278 = fract(_1235);
            float _1247 = _1278.x;
            float _1248 = 1.0 - _1247;
            float _1250 = _1278.y;
            float _1251 = 1.0 - _1250;
            spvUnsafeArray<float, 4> _2010 = spvUnsafeArray<float, 4>({ _1248 * _1251, _1247 * _1251, _1248 * _1250, _1247 * _1250 });
            spvUnsafeArray<float, 4> _1986 = _2010;
            float4 _2128;
            _2128 = float4(0.0);
            float4 _2146;
            for (int _2127 = 0; _2127 < 4; _2128 = _2146, _2127++)
            {
                int2 _1193 = _1238 + _2150[_2127];
                if (all(uint2(_1193) < uint2(_1114)))
                {
                    float4 _1293 = r_inpainting_pyramid.read(uint2(_1193), _2069);
                    float _1207 = _1986[_2127] * float(_1293.w > 0.0);
                    _2146 = _2128 + float4(_1293.xyz * _1207, _1207);
                }
                else
                {
                    _2146 = _2128;
                }
            }
            if (_2128.w > 0.0)
            {
                _2133 = _2070 + (float4(_2128.xyz / float3(_2128.w), 1.0) * (powr(1.0 - (float(_2069) * 0.100000001490116119384765625), 3.0) * _2128.w));
            }
            else
            {
                _2133 = _2070;
            }
        }
        float3 _1309 = mix(_1085.xyz, (_2070.xyz / float3(_2070.w)) * float(cbFI.displaySize.x > 0), float3(_975));
        float4 _2026;
        _2026.x = _1309.x;
        _2026.y = _1309.y;
        _2026.z = _1309.z;
        _2088 = _2026;
    }
    else
    {
        _2088 = _1085;
    }
    float4 _2089;
    bool _2092;
    if (cbFI.HUDLessAttachedFactor == 1)
    {
        float3 _1319 = r_current_interpolation_source.read(uint2(_922), 0).xyz;
        float3 _1004 = r_present_backbuffer.read(uint2(_922), 0).xyz;
        float4 _2090;
        bool _2094;
        if (any(abs(_1319 - _1004) > float3(0.0)))
        {
            float3 _2072;
            switch (cbFI.backBufferTransferFunction)
            {
                case 0u:
                {
                    float3 _1407 = fast::clamp((_1319 - float3(0.040449999272823333740234375)) * float3(as_type<float>(0xff800000u /* -inf */)), float3(0.0), float3(1.0));
                    float3 _1380 = powr((_1319 * float3(0.94786727428436279296875)) + float3(0.0521326996386051177978515625), float3(2.400000095367431640625));
                    _2072 = (_1407 * (_1319 * float3(0.077399380505084991455078125))) + (((-_1407) * _1380) + _1380);
                    break;
                }
                case 1u:
                {
                    float3 _1441 = powr(_1319, float3(0.01268330030143260955810546875));
                    _2072 = powr(fast::clamp(_1441 - float3(0.835937976837158203125), float3(0.0), float3(1.0)) / (float3(18.85160064697265625) - (float3(18.6875) * _1441)), float3(6.277390003204345703125)) * (10000.0 / cbFI.minMaxLuminance.y);
                    break;
                }
                case 2u:
                {
                    _2072 = (_1319 - float3(cbFI.minMaxLuminance.x * 0.012500000186264514923095703125)) / float3((cbFI.minMaxLuminance.y - cbFI.minMaxLuminance.x) * 0.012500000186264514923095703125);
                    break;
                }
                default:
                {
                    _2072 = _2074;
                    break;
                }
            }
            float3 _2076;
            switch (cbFI.backBufferTransferFunction)
            {
                case 0u:
                {
                    float3 _1609 = fast::clamp((_1004 - float3(0.040449999272823333740234375)) * float3(as_type<float>(0xff800000u /* -inf */)), float3(0.0), float3(1.0));
                    float3 _1582 = powr((_1004 * float3(0.94786727428436279296875)) + float3(0.0521326996386051177978515625), float3(2.400000095367431640625));
                    _2076 = (_1609 * (_1004 * float3(0.077399380505084991455078125))) + (((-_1609) * _1582) + _1582);
                    break;
                }
                case 1u:
                {
                    float3 _1643 = powr(_1004, float3(0.01268330030143260955810546875));
                    _2076 = powr(fast::clamp(_1643 - float3(0.835937976837158203125), float3(0.0), float3(1.0)) / (float3(18.85160064697265625) - (float3(18.6875) * _1643)), float3(6.277390003204345703125)) * (10000.0 / cbFI.minMaxLuminance.y);
                    break;
                }
                case 2u:
                {
                    _2076 = (_1004 - float3(cbFI.minMaxLuminance.x * 0.012500000186264514923095703125)) / float3((cbFI.minMaxLuminance.y - cbFI.minMaxLuminance.x) * 0.012500000186264514923095703125);
                    break;
                }
                default:
                {
                    _2076 = _2074;
                    break;
                }
            }
            float _1792 = fast::max(_2072.x, _2076.x);
            float _2080;
            if (_1792 != 0.0)
            {
                _2080 = fast::min(_2072.x, _2076.x) / _1792;
            }
            else
            {
                _2080 = 0.0;
            }
            float _1824 = fast::max(_2072.y, _2076.y);
            float _2081;
            if (_1824 != 0.0)
            {
                _2081 = fast::min(_2072.y, _2076.y) / _1824;
            }
            else
            {
                _2081 = 0.0;
            }
            float _1856 = fast::max(_2072.z, _2076.z);
            float _2082;
            if (_1856 != 0.0)
            {
                _2082 = fast::min(_2072.z, _2076.z) / _1856;
            }
            else
            {
                _2082 = 0.0;
            }
            float3 _1873 = fast::clamp(float3(fast::clamp((1.0 - _2080) * 10.0, 0.0, 1.0), fast::clamp((1.0 - _2081) * 10.0, 0.0, 1.0), fast::clamp((1.0 - _2082) * 10.0, 0.0, 1.0)), float3(0.0), float3(1.0));
            float _1769 = fast::max(_1873.x, fast::max(_1873.y, _1873.z));
            bool _1018 = _1769 > 0.001000000047497451305389404296875;
            float4 _2091;
            if (_1018)
            {
                float3 _1888 = mix(_2088.xyz, _1004, float3(_1769));
                float4 _2041;
                _2041.x = _1888.x;
                _2041.y = _1888.y;
                _2041.z = _1888.z;
                _2091 = _2041;
            }
            else
            {
                _2091 = _2088;
            }
            _2094 = _1018 ? true : _977;
            _2090 = _2091;
        }
        else
        {
            _2094 = _977;
            _2090 = _2088;
        }
        _2092 = _2094;
        _2089 = _2090;
    }
    else
    {
        _2092 = _977;
        _2089 = _2088;
    }
    float4 _2107;
    bool _2108;
    if ((cbFI.dispatchFlags & 1u) != 0u)
    {
        float3 _1040 = _2089.xyz;
        int _1895 = _922.x;
        float3 _2101;
        bool _2105;
        if (_1895 < 16)
        {
            float3 _2048 = _1040;
            _2048.y = 1.0;
            _2105 = true;
            _2101 = _2048;
        }
        else
        {
            bool _1905 = _1895 > (cbFI.displaySize.x - 16);
            float3 _2102;
            if (_1905)
            {
                _2102 = _1040 + float3(cbFI.debugBarColor);
            }
            else
            {
                _2102 = _1040;
            }
            _2105 = _1905 ? true : _2092;
            _2101 = _2102;
        }
        float4 _2051;
        _2051.x = _2101.x;
        _2051.y = _2101.y;
        _2051.z = _2101.z;
        _2108 = _2105;
        _2107 = _2051;
    }
    else
    {
        _2108 = _2092;
        _2107 = _2089;
    }
    bool _2124;
    float4 _2125;
    if ((cbFI.dispatchFlags & 2u) != 0u)
    {
        float3 _1058 = _2107.xyz;
        int _1926 = _922.y;
        bool _1927 = _1926 < 32;
        bool _1931;
        if (_1927)
        {
            _1931 = cbFI.reset == 1;
        }
        else
        {
            _1931 = _1927;
        }
        float3 _2112;
        bool _2119;
        if (_1931)
        {
            float3 _2058 = _1058;
            _2058.x = 1.0;
            _2119 = true;
            _2112 = _2058;
        }
        else
        {
            bool _1937 = _1926 > 32;
            bool _1943;
            if (_1937)
            {
                _1943 = _1926 < 64;
            }
            else
            {
                _1943 = _1937;
            }
            bool _1947;
            if (_1943)
            {
                _1947 = (r_optical_flow_scd.read(uint2(int2(1, 0)), 0).x & 15u) != 0u;
            }
            else
            {
                _1947 = _1943;
            }
            float3 _2113;
            if (_1947)
            {
                float3 _2062 = _1058;
                _2062.z = 1.0;
                _2113 = _2062;
            }
            else
            {
                _2113 = _1058;
            }
            _2119 = _1947 ? true : _2108;
            _2112 = _2113;
        }
        float4 _2064;
        _2064.x = _2112.x;
        _2064.y = _2112.y;
        _2064.z = _2112.z;
        _2125 = _2064;
        _2124 = _2119;
    }
    else
    {
        _2125 = _2107;
        _2124 = _2108;
    }
    if (_2124)
    {
        rw_output.write(float4(_2125.xyz, 1.0), uint2(_922));
    }
}


}

// ===== AMD FSR optical_flow_vector_field =====
namespace ism_optical_flow_vector_field {
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wmissing-braces"


using namespace metal;

template<typename T, size_t Num>
struct spvUnsafeArray
{
    T elements[Num ? Num : 1];
    
    thread T& operator [] (size_t pos) thread
    {
        return elements[pos];
    }
    constexpr const thread T& operator [] (size_t pos) const thread
    {
        return elements[pos];
    }
    
    device T& operator [] (size_t pos) device
    {
        return elements[pos];
    }
    constexpr const device T& operator [] (size_t pos) const device
    {
        return elements[pos];
    }
    
    constexpr const constant T& operator [] (size_t pos) const constant
    {
        return elements[pos];
    }
    
    threadgroup T& operator [] (size_t pos) threadgroup
    {
        return elements[pos];
    }
    constexpr const threadgroup T& operator [] (size_t pos) const threadgroup
    {
        return elements[pos];
    }
};

struct cbFI_t
{
    int2 renderSize;
    int2 displaySize;
    float2 displaySizeRcp;
    float cameraNear;
    float cameraFar;
    int2 upscalerTargetSize;
    int Mode;
    int reset;
    float4 fDeviceToViewDepth;
    float deltaTime;
    int HUDLessAttachedFactor;
    int2 distortionFieldSize;
    float2 opticalFlowScale;
    int opticalFlowBlockSize;
    uint dispatchFlags;
    int2 maxRenderSize;
    int opticalFlowHalfResMode;
    int NumInstances;
    int2 interpolationRectBase;
    int2 interpolationRectSize;
    packed_float3 debugBarColor;
    uint backBufferTransferFunction;
    float2 minMaxLuminance;
    float fTanHalfFOV;
    int _pad1;
    float2 fJitter;
    float2 fMotionVectorScale;
};

constant uint3 gl_WorkGroupSize [[maybe_unused]] = uint3(8u, 8u, 1u);
constant float3 _3020 = {};

constant spvUnsafeArray<int2, 4> _3096 = spvUnsafeArray<int2, 4>({ int2(0), int2(1, 0), int2(0, 1), int2(1) });

[[host_name("ism_fsr_optical_flow_vector_field")]] kernel void ism_fsr_optical_flow_vector_field(constant cbFI_t& cbFI [[buffer(0)]], texture2d<float> r_previous_interpolation_source [[texture(0)]], texture2d<float> r_current_interpolation_source [[texture(1)]], texture2d<int> r_optical_flow [[texture(2)]], texture2d<uint, access::read_write> rw_optical_flow_motion_vector_field_x [[texture(3)]], texture2d<uint, access::read_write> rw_optical_flow_motion_vector_field_y [[texture(4)]], sampler s_LinearClamp [[sampler(0)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    int2 _986 = int2(gl_GlobalInvocationID.xy);
    int _3008;
    float _3009;
    float2 _3010;
    _3010 = float2(0.0);
    _3009 = 0.0;
    _3008 = -1;
    float2 _3066;
    float _3067;
    for (; _3008 <= 1; _3010 = _3066, _3009 = _3067, _3008++)
    {
        _3067 = _3009;
        _3066 = _3010;
        for (int _3064 = -1; _3064 <= 1; )
        {
            float2 _1179 = float2(r_optical_flow.read(uint2((_986 + int2(_3064, _3008))), 0).xy) * cbFI.opticalFlowScale;
            float _1069 = length(_1179 * float2(cbFI.interpolationRectSize));
            float _1076 = fast::max(0.0, 512.0 - _1069) * float(_1069 > 1.0);
            _3067 += _1076;
            _3066 += (_1179 * _1076);
            _3064++;
            continue;
        }
    }
    float2 _1097 = _3010 / float2(_3009);
    int _3011;
    float _3012;
    float2 _3013;
    _3013 = float2(0.0);
    _3012 = 0.0;
    _3011 = -1;
    float2 _3062;
    float _3063;
    for (; _3011 <= 1; _3013 = _3062, _3012 = _3063, _3011++)
    {
        _3063 = _3012;
        _3062 = _3013;
        for (int _3058 = -1; _3058 <= 1; )
        {
            float2 _1213 = float2(r_optical_flow.read(uint2((_986 + int2(_3058, _3011))), 0).xy) * cbFI.opticalFlowScale;
            float _1128 = length(_1213 * float2(cbFI.interpolationRectSize));
            float _1142 = fast::max(0.0, powr(dot(_1097, _1213), 1.25)) * (fast::max(0.0, 512.0 - _1128) * float(_1128 > 1.0));
            _3063 += _1142;
            _3062 += (_1213 * _1142);
            _3058++;
            continue;
        }
    }
    float2 _3015;
    if (_3012 > 0.001000000047497451305389404296875)
    {
        _3015 = _3013 / float2(_3012);
    }
    else
    {
        _3015 = _3013;
    }
    int2 _1376 = int2((float2(1.0) / cbFI.opticalFlowScale) / float2(int2(cbFI.opticalFlowBlockSize))) * int2(1);
    float2 _1288 = float2(_1376);
    float2 _1289 = (float2(gl_GlobalInvocationID.xy) + float2(0.5)) / _1288;
    float2 _1291 = _3015 * 0.5;
    float2 _1294 = _1289 + _3015;
    float3 _1398 = r_previous_interpolation_source.sample(s_LinearClamp, _1294, level(0.0)).xyz;
    float3 _1406 = r_current_interpolation_source.sample(s_LinearClamp, _1289, level(0.0)).xyz;
    float _3029;
    switch (cbFI.backBufferTransferFunction)
    {
        case 0u:
        {
            float3 _3025;
            switch (cbFI.backBufferTransferFunction)
            {
                case 0u:
                {
                    float3 _1517 = fast::clamp((_1398 - float3(0.040449999272823333740234375)) * float3(as_type<float>(0xff800000u /* -inf */)), float3(0.0), float3(1.0));
                    float3 _1490 = powr((_1398 * float3(0.94786727428436279296875)) + float3(0.0521326996386051177978515625), float3(2.400000095367431640625));
                    _3025 = (_1517 * (_1398 * float3(0.077399380505084991455078125))) + (((-_1517) * _1490) + _1490);
                    break;
                }
                case 1u:
                {
                    float3 _1551 = powr(_1398, float3(0.01268330030143260955810546875));
                    _3025 = powr(fast::clamp(_1551 - float3(0.835937976837158203125), float3(0.0), float3(1.0)) / (float3(18.85160064697265625) - (float3(18.6875) * _1551)), float3(6.277390003204345703125)) * (10000.0 / cbFI.minMaxLuminance.y);
                    break;
                }
                case 2u:
                {
                    _3025 = (_1398 - float3(cbFI.minMaxLuminance.x * 0.012500000186264514923095703125)) / float3((cbFI.minMaxLuminance.y - cbFI.minMaxLuminance.x) * 0.012500000186264514923095703125);
                    break;
                }
                default:
                {
                    _3025 = _3020;
                    break;
                }
            }
            _3029 = dot(_3025, float3(0.2125999927520751953125, 0.715200006961822509765625, 0.072200000286102294921875));
            break;
        }
        case 1u:
        {
            float3 _3021;
            switch (cbFI.backBufferTransferFunction)
            {
                case 0u:
                {
                    float3 _1723 = fast::clamp((_1398 - float3(0.040449999272823333740234375)) * float3(as_type<float>(0xff800000u /* -inf */)), float3(0.0), float3(1.0));
                    float3 _1696 = powr((_1398 * float3(0.94786727428436279296875)) + float3(0.0521326996386051177978515625), float3(2.400000095367431640625));
                    _3021 = (_1723 * (_1398 * float3(0.077399380505084991455078125))) + (((-_1723) * _1696) + _1696);
                    break;
                }
                case 1u:
                {
                    float3 _1757 = powr(_1398, float3(0.01268330030143260955810546875));
                    _3021 = powr(fast::clamp(_1757 - float3(0.835937976837158203125), float3(0.0), float3(1.0)) / (float3(18.85160064697265625) - (float3(18.6875) * _1757)), float3(6.277390003204345703125)) * (10000.0 / cbFI.minMaxLuminance.y);
                    break;
                }
                case 2u:
                {
                    _3021 = (_1398 - float3(cbFI.minMaxLuminance.x * 0.012500000186264514923095703125)) / float3((cbFI.minMaxLuminance.y - cbFI.minMaxLuminance.x) * 0.012500000186264514923095703125);
                    break;
                }
                default:
                {
                    _3021 = _3020;
                    break;
                }
            }
            _3029 = ((0.2626999914646148681640625 * _3021.x) + (0.677999973297119140625 * _3021.y)) + (0.0593000017106533050537109375 * _3021.z);
            break;
        }
        case 2u:
        {
            float3 _3016;
            switch (cbFI.backBufferTransferFunction)
            {
                case 0u:
                {
                    float3 _1940 = fast::clamp((_1398 - float3(0.040449999272823333740234375)) * float3(as_type<float>(0xff800000u /* -inf */)), float3(0.0), float3(1.0));
                    float3 _1913 = powr((_1398 * float3(0.94786727428436279296875)) + float3(0.0521326996386051177978515625), float3(2.400000095367431640625));
                    _3016 = (_1940 * (_1398 * float3(0.077399380505084991455078125))) + (((-_1940) * _1913) + _1913);
                    break;
                }
                case 1u:
                {
                    float3 _1974 = powr(_1398, float3(0.01268330030143260955810546875));
                    _3016 = powr(fast::clamp(_1974 - float3(0.835937976837158203125), float3(0.0), float3(1.0)) / (float3(18.85160064697265625) - (float3(18.6875) * _1974)), float3(6.277390003204345703125)) * (10000.0 / cbFI.minMaxLuminance.y);
                    break;
                }
                case 2u:
                {
                    _3016 = (_1398 - float3(cbFI.minMaxLuminance.x * 0.012500000186264514923095703125)) / float3((cbFI.minMaxLuminance.y - cbFI.minMaxLuminance.x) * 0.012500000186264514923095703125);
                    break;
                }
                default:
                {
                    _3016 = _3020;
                    break;
                }
            }
            _3029 = dot(_3016, float3(0.2125999927520751953125, 0.715200006961822509765625, 0.072200000286102294921875));
            break;
        }
        default:
        {
            _3029 = 0.0;
            break;
        }
    }
    float _1300 = 0.001000000047497451305389404296875 + _3029;
    float _3054;
    switch (cbFI.backBufferTransferFunction)
    {
        case 0u:
        {
            float3 _3046;
            switch (cbFI.backBufferTransferFunction)
            {
                case 0u:
                {
                    float3 _2174 = fast::clamp((_1406 - float3(0.040449999272823333740234375)) * float3(as_type<float>(0xff800000u /* -inf */)), float3(0.0), float3(1.0));
                    float3 _2147 = powr((_1406 * float3(0.94786727428436279296875)) + float3(0.0521326996386051177978515625), float3(2.400000095367431640625));
                    _3046 = (_2174 * (_1406 * float3(0.077399380505084991455078125))) + (((-_2174) * _2147) + _2147);
                    break;
                }
                case 1u:
                {
                    float3 _2208 = powr(_1406, float3(0.01268330030143260955810546875));
                    _3046 = powr(fast::clamp(_2208 - float3(0.835937976837158203125), float3(0.0), float3(1.0)) / (float3(18.85160064697265625) - (float3(18.6875) * _2208)), float3(6.277390003204345703125)) * (10000.0 / cbFI.minMaxLuminance.y);
                    break;
                }
                case 2u:
                {
                    _3046 = (_1406 - float3(cbFI.minMaxLuminance.x * 0.012500000186264514923095703125)) / float3((cbFI.minMaxLuminance.y - cbFI.minMaxLuminance.x) * 0.012500000186264514923095703125);
                    break;
                }
                default:
                {
                    _3046 = _3020;
                    break;
                }
            }
            _3054 = dot(_3046, float3(0.2125999927520751953125, 0.715200006961822509765625, 0.072200000286102294921875));
            break;
        }
        case 1u:
        {
            float3 _3038;
            switch (cbFI.backBufferTransferFunction)
            {
                case 0u:
                {
                    float3 _2380 = fast::clamp((_1406 - float3(0.040449999272823333740234375)) * float3(as_type<float>(0xff800000u /* -inf */)), float3(0.0), float3(1.0));
                    float3 _2353 = powr((_1406 * float3(0.94786727428436279296875)) + float3(0.0521326996386051177978515625), float3(2.400000095367431640625));
                    _3038 = (_2380 * (_1406 * float3(0.077399380505084991455078125))) + (((-_2380) * _2353) + _2353);
                    break;
                }
                case 1u:
                {
                    float3 _2414 = powr(_1406, float3(0.01268330030143260955810546875));
                    _3038 = powr(fast::clamp(_2414 - float3(0.835937976837158203125), float3(0.0), float3(1.0)) / (float3(18.85160064697265625) - (float3(18.6875) * _2414)), float3(6.277390003204345703125)) * (10000.0 / cbFI.minMaxLuminance.y);
                    break;
                }
                case 2u:
                {
                    _3038 = (_1406 - float3(cbFI.minMaxLuminance.x * 0.012500000186264514923095703125)) / float3((cbFI.minMaxLuminance.y - cbFI.minMaxLuminance.x) * 0.012500000186264514923095703125);
                    break;
                }
                default:
                {
                    _3038 = _3020;
                    break;
                }
            }
            _3054 = ((0.2626999914646148681640625 * _3038.x) + (0.677999973297119140625 * _3038.y)) + (0.0593000017106533050537109375 * _3038.z);
            break;
        }
        case 2u:
        {
            float3 _3030;
            switch (cbFI.backBufferTransferFunction)
            {
                case 0u:
                {
                    float3 _2597 = fast::clamp((_1406 - float3(0.040449999272823333740234375)) * float3(as_type<float>(0xff800000u /* -inf */)), float3(0.0), float3(1.0));
                    float3 _2570 = powr((_1406 * float3(0.94786727428436279296875)) + float3(0.0521326996386051177978515625), float3(2.400000095367431640625));
                    _3030 = (_2597 * (_1406 * float3(0.077399380505084991455078125))) + (((-_2597) * _2570) + _2570);
                    break;
                }
                case 1u:
                {
                    float3 _2631 = powr(_1406, float3(0.01268330030143260955810546875));
                    _3030 = powr(fast::clamp(_2631 - float3(0.835937976837158203125), float3(0.0), float3(1.0)) / (float3(18.85160064697265625) - (float3(18.6875) * _2631)), float3(6.277390003204345703125)) * (10000.0 / cbFI.minMaxLuminance.y);
                    break;
                }
                case 2u:
                {
                    _3030 = (_1406 - float3(cbFI.minMaxLuminance.x * 0.012500000186264514923095703125)) / float3((cbFI.minMaxLuminance.y - cbFI.minMaxLuminance.x) * 0.012500000186264514923095703125);
                    break;
                }
                default:
                {
                    _3030 = _3020;
                    break;
                }
            }
            _3054 = dot(_3030, float3(0.2125999927520751953125, 0.715200006961822509765625, 0.072200000286102294921875));
            break;
        }
        default:
        {
            _3054 = 0.0;
            break;
        }
    }
    float _1303 = 0.001000000047497451305389404296875 + _3054;
    float2 _1306 = float2(cbFI.interpolationRectSize);
    float _1308 = length(_3015 * _1306);
    uint _1321 = uint(_1308 > 1.0) * uint(fast::clamp(_1308 / length(_1306 * 0.0500000007450580596923828125), 0.0, 1.0) * 1023.0);
    if (_1321 > 0u)
    {
        float _2755 = fast::max(_1300, _1303);
        float _3055;
        if (_2755 != 0.0)
        {
            _3055 = fast::min(_1300, _1303) / _2755;
        }
        else
        {
            _3055 = 0.0;
        }
        float _2773 = _1294.x;
        bool _2774 = _2773 > 0.0;
        bool _2780;
        if (_2774)
        {
            _2780 = _2773 < 1.0;
        }
        else
        {
            _2780 = _2774;
        }
        bool _2792;
        if (_2780)
        {
            float _2783 = _1294.y;
            bool _2784 = _2783 > 0.0;
            bool _2790;
            if (_2784)
            {
                _2790 = _2783 < 1.0;
            }
            else
            {
                _2790 = _2784;
            }
            _2792 = _2790;
        }
        else
        {
            _2792 = _2780;
        }
        uint _2810 = (2147483648u | ((_1321 & 1023u) << 21u)) | (((uint(rint(powr(_3055, 1.0) * 31.0)) * uint(_2792)) & 31u) << 16u);
        uint _2815 = _2810 | as_type<uint>(half2(float2(_1291.x, 0.0)));
        uint _2820 = _2810 | as_type<uint>(half2(float2(_1291.y, 0.0)));
        int2 _2866 = int2(floor(((_1289 + _1291) * _1288) - float2(0.5)));
        for (int _3056 = 0; _3056 < 4; _3056++)
        {
            int2 _1358 = _2866 + _3096[_3056];
            if (all(uint2(_1358) < uint2(_1376)))
            {
                uint _2939 = rw_optical_flow_motion_vector_field_x.atomic_fetch_max(uint2(_1358), _2815).x;
                uint _2944 = rw_optical_flow_motion_vector_field_y.atomic_fetch_max(uint2(_1358), _2820).x;
            }
        }
    }
}


}

// ===== AMD FSR interpolation =====
namespace ism_interpolation {
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wmissing-braces"


using namespace metal;

template<typename T, size_t Num>
struct spvUnsafeArray
{
    T elements[Num ? Num : 1];
    
    thread T& operator [] (size_t pos) thread
    {
        return elements[pos];
    }
    constexpr const thread T& operator [] (size_t pos) const thread
    {
        return elements[pos];
    }
    
    device T& operator [] (size_t pos) device
    {
        return elements[pos];
    }
    constexpr const device T& operator [] (size_t pos) const device
    {
        return elements[pos];
    }
    
    constexpr const constant T& operator [] (size_t pos) const constant
    {
        return elements[pos];
    }
    
    threadgroup T& operator [] (size_t pos) threadgroup
    {
        return elements[pos];
    }
    constexpr const threadgroup T& operator [] (size_t pos) const threadgroup
    {
        return elements[pos];
    }
};

struct cbFI_t
{
    int2 renderSize;
    int2 displaySize;
    float2 displaySizeRcp;
    float cameraNear;
    float cameraFar;
    int2 upscalerTargetSize;
    int Mode;
    int reset;
    float4 fDeviceToViewDepth;
    float deltaTime;
    int HUDLessAttachedFactor;
    int2 distortionFieldSize;
    float2 opticalFlowScale;
    int opticalFlowBlockSize;
    uint dispatchFlags;
    int2 maxRenderSize;
    int opticalFlowHalfResMode;
    int NumInstances;
    int2 interpolationRectBase;
    int2 interpolationRectSize;
    packed_float3 debugBarColor;
    uint backBufferTransferFunction;
    float2 minMaxLuminance;
    float fTanHalfFOV;
    int _pad1;
    float2 fJitter;
    float2 fMotionVectorScale;
};

struct FrameInterpolationCounters_t
{
    uint data[1];
};

constant uint3 gl_WorkGroupSize [[maybe_unused]] = uint3(8u, 8u, 1u);

constant spvUnsafeArray<int2, 4> _5710 = spvUnsafeArray<int2, 4>({ int2(0), int2(1, 0), int2(0, 1), int2(1) });

[[host_name("ism_fsr_interpolation")]] kernel void ism_fsr_interpolation(constant cbFI_t& cbFI [[buffer(0)]], const device FrameInterpolationCounters_t& r_counters [[buffer(1)]], texture2d<float> r_previous_interpolation_source [[texture(0)]], texture2d<float> r_current_interpolation_source [[texture(1)]], texture2d<float> r_disocclusion_mask [[texture(2)]], texture2d<uint> r_game_motion_vector_field_x [[texture(3)]], texture2d<uint> r_game_motion_vector_field_y [[texture(4)]], texture2d<uint> r_optical_flow_motion_vector_field_x [[texture(5)]], texture2d<uint> r_optical_flow_motion_vector_field_y [[texture(6)]], texture2d<float> r_inpainting_pyramid [[texture(7)]], texture2d<float, access::write> rw_output [[texture(8)]], sampler s_LinearClamp [[sampler(0)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    int2 _1558 = int2(gl_GlobalInvocationID.xy);
    int _1623 = _1558.x;
    bool _1626 = _1623 >= cbFI.interpolationRectBase.x;
    bool _1637;
    if (_1626)
    {
        _1637 = _1623 < (cbFI.interpolationRectSize.x + cbFI.interpolationRectBase.x);
    }
    else
    {
        _1637 = _1626;
    }
    bool _1645;
    if (_1637)
    {
        _1645 = _1558.y >= cbFI.interpolationRectBase.y;
    }
    else
    {
        _1645 = _1637;
    }
    bool _1656;
    if (_1645)
    {
        _1656 = _1558.y < (cbFI.interpolationRectSize.y + cbFI.interpolationRectBase.y);
    }
    else
    {
        _1656 = _1645;
    }
    bool _1592;
    if (_1656)
    {
        _1592 = r_counters.data[1] == 0u;
    }
    else
    {
        _1592 = !_1656;
    }
    float3 _5569;
    float _5570;
    if (_1592)
    {
        _5570 = 0.0;
        _5569 = r_current_interpolation_source.read(uint2(_1558), 0).xyz;
    }
    else
    {
        float2 _1752 = float2(cbFI.interpolationRectSize);
        float2 _1753 = (float2(gl_GlobalInvocationID.xy - uint2(cbFI.interpolationRectBase)) + float2(0.5)) / _1752;
        float2 _1759 = float2(cbFI.displaySize);
        float2 _1760 = (float2(gl_GlobalInvocationID.xy) + float2(0.5)) / _1759;
        float2 _1763 = float2(cbFI.renderSize);
        float2 _1772 = _1752 / _1759;
        int2 _1990 = int2(_1753 * _1763);
        uint4 _2062 = r_game_motion_vector_field_x.read(uint2(_1990), 0);
        uint _2063 = _2062.x;
        float _2080 = float((_2063 >> 21u) & 1023u) * 0.000977517105638980865478515625;
        float2 _5206;
        if ((isunordered(_2080, 0.0) || _2080 <= 0.0))
        {
            float4 _5204;
            int2 _5562;
            _5562 = cbFI.renderSize;
            _5204 = float4(0.0);
            int2 _2014;
            float4 _2213;
            int _5203 = 0;
            for (;;)
            {
                bool _2004 = _5203 < 11;
                bool _2010;
                if (_2004)
                {
                    _2010 = _5204.w == 0.0;
                }
                else
                {
                    _2010 = _2004;
                }
                if (_2010)
                {
                    _2014 = _5562 / int2(2);
                    float2 _2225 = (_1753 * float2(_2014)) - float2(0.5);
                    int2 _2228 = int2(floor(_2225));
                    float2 _2268 = fract(_2225);
                    float _2237 = _2268.x;
                    float _2238 = 1.0 - _2237;
                    float _2240 = _2268.y;
                    float _2241 = 1.0 - _2240;
                    spvUnsafeArray<float, 4> _5020 = spvUnsafeArray<float, 4>({ _2238 * _2241, _2237 * _2241, _2238 * _2240, _2237 * _2240 });
                    spvUnsafeArray<float, 4> _4814 = _5020;
                    float _5565;
                    float4 _5566;
                    _5566 = float4(0.0);
                    _5565 = 0.0;
                    float _5707;
                    float4 _5708;
                    for (int _5564 = 0; _5564 < 4; _5566 = _5708, _5565 = _5707, _5564++)
                    {
                        int2 _2174 = _2228 + _5710[_5564];
                        if (all(uint2(_2174) < uint2(_2014)))
                        {
                            float4 _2283 = r_inpainting_pyramid.read(uint2(_2174), _5203);
                            float _2182 = _2283.z;
                            float _2193 = (_4814[_5564] * float(_2182 > 0.0)) * _2182;
                            _5708 = _5566 + (_2283 * _2193);
                            _5707 = _5565 + _2193;
                        }
                        else
                        {
                            _5708 = _5566;
                            _5707 = _5565;
                        }
                    }
                    _2213 = _5566 / float4((_5565 > 0.0) ? _5565 : 1.0);
                    _5562 = _2014;
                    _5204 = _2213;
                    _5203++;
                    continue;
                }
                else
                {
                    break;
                }
            }
            _5206 = _5204.xy;
        }
        else
        {
            _5206 = float2(float2(as_type<half2>(_2063)).x, float2(as_type<half2>(r_game_motion_vector_field_y.read(uint2(_1990), 0).x)).x);
        }
        float2 _2037 = _1753 - _5206;
        float _2287 = _2037.x;
        bool _2288 = _2287 > 0.0;
        bool _2294;
        if (_2288)
        {
            _2294 = _2287 < 1.0;
        }
        else
        {
            _2294 = _2288;
        }
        bool _2306;
        if (_2294)
        {
            float _2297 = _2037.y;
            bool _2298 = _2297 > 0.0;
            bool _2304;
            if (_2298)
            {
                _2304 = _2297 < 1.0;
            }
            else
            {
                _2304 = _2298;
            }
            _2306 = _2304;
        }
        else
        {
            _2306 = _2294;
        }
        float2 _2044 = _1753 + _5206;
        float _2310 = _2044.x;
        bool _2311 = _2310 > 0.0;
        bool _2317;
        if (_2311)
        {
            _2317 = _2310 < 1.0;
        }
        else
        {
            _2317 = _2311;
        }
        bool _2329;
        if (_2317)
        {
            float _2320 = _2044.y;
            bool _2321 = _2320 > 0.0;
            bool _2327;
            if (_2321)
            {
                _2327 = _2320 < 1.0;
            }
            else
            {
                _2327 = _2321;
            }
            _2329 = _2327;
        }
        else
        {
            _2329 = _2317;
        }
        int2 _2351 = int2(float2(int2((float2(1.0) / cbFI.opticalFlowScale) / float2(int2(cbFI.opticalFlowBlockSize))) * int2(1)) * 1.0);
        float2 _2484 = (_1760 * float2(_2351)) - float2(0.5);
        int2 _2487 = int2(floor(_2484));
        float2 _2527 = fract(_2484);
        float _2496 = _2527.x;
        float _2497 = 1.0 - _2496;
        float _2499 = _2527.y;
        float _2500 = 1.0 - _2499;
        spvUnsafeArray<float, 4> _5002 = spvUnsafeArray<float, 4>({ _2497 * _2500, _2496 * _2500, _2497 * _2499, _2496 * _2499 });
        spvUnsafeArray<float, 4> _4763 = _5002;
        float _5238;
        float2 _5239;
        _5239 = float2(0.0);
        _5238 = 0.0;
        float _5571;
        float2 _5572;
        for (int _5237 = 0; _5237 < 4; _5239 = _5572, _5238 = _5571, _5237++)
        {
            int2 _2366 = _2487 + _5710[_5237];
            if (all(uint2(_2366) < uint2(_2351)))
            {
                _5572 = _5239 + (float2(float2(as_type<half2>(r_optical_flow_motion_vector_field_x.read(uint2(_2366), 0).x)).x, float2(as_type<half2>(r_optical_flow_motion_vector_field_y.read(uint2(_2366), 0).x)).x) * _4763[_5237]);
                _5571 = _5238 + _4763[_5237];
            }
            else
            {
                _5572 = _5239;
                _5571 = _5238;
            }
        }
        float2 _5242;
        if (_5238 > 0.0)
        {
            _5242 = _5239 / float2(_5238);
        }
        else
        {
            _5242 = _5239;
        }
        float2 _1784 = float2(fast::clamp(r_disocclusion_mask.sample(s_LinearClamp, (_1753 * (_1763 / float2(cbFI.maxRenderSize))), level(0.0)).xy, float2(0.0), float2(1.0)) == float2(1.0));
        float2 _2844 = ((_1760 + (_5206 * _1772)) * _1759) - float2(0.5);
        int2 _2847 = int2(floor(_2844));
        float2 _2887 = fract(_2844);
        float _2856 = _2887.x;
        float _2857 = 1.0 - _2856;
        float _2859 = _2887.y;
        float _2860 = 1.0 - _2859;
        spvUnsafeArray<float, 4> _4984 = spvUnsafeArray<float, 4>({ _2857 * _2860, _2856 * _2860, _2857 * _2859, _2856 * _2859 });
        spvUnsafeArray<float, 4> _4698 = _4984;
        float _5267;
        float3 _5268;
        _5268 = float3(0.0);
        _5267 = 0.0;
        float _5575;
        float3 _5576;
        for (int _5266 = 0; _5266 < 4; _5268 = _5576, _5267 = _5575, _5266++)
        {
            int2 _2776 = _2847 + _5710[_5266];
            int _2899 = _2776.x;
            bool _2902 = _2899 >= cbFI.interpolationRectBase.x;
            bool _2913;
            if (_2902)
            {
                _2913 = _2899 < (cbFI.interpolationRectSize.x + cbFI.interpolationRectBase.x);
            }
            else
            {
                _2913 = _2902;
            }
            bool _2921;
            if (_2913)
            {
                _2921 = _2776.y >= cbFI.interpolationRectBase.y;
            }
            else
            {
                _2921 = _2913;
            }
            bool _2932;
            if (_2921)
            {
                _2932 = _2776.y < (cbFI.interpolationRectSize.y + cbFI.interpolationRectBase.y);
            }
            else
            {
                _2932 = _2921;
            }
            if (_2932)
            {
                _5576 = _5268 + (r_previous_interpolation_source.read(uint2(_2776), 0).xyz * _4698[_5266]);
                _5575 = _5267 + _4698[_5266];
            }
            else
            {
                _5576 = _5268;
                _5575 = _5267;
            }
        }
        float3 _5269;
        if (_5267 != 0.0)
        {
            _5269 = _5268 / float3(_5267);
        }
        else
        {
            _5269 = float3(0.0);
        }
        float2 _3257 = ((_1760 + ((-_5206) * _1772)) * _1759) - float2(0.5);
        int2 _3260 = int2(floor(_3257));
        float2 _3300 = fract(_3257);
        float _3269 = _3300.x;
        float _3270 = 1.0 - _3269;
        float _3272 = _3300.y;
        float _3273 = 1.0 - _3272;
        spvUnsafeArray<float, 4> _4966 = spvUnsafeArray<float, 4>({ _3270 * _3273, _3269 * _3273, _3270 * _3272, _3269 * _3272 });
        spvUnsafeArray<float, 4> _4658 = _4966;
        float _5301;
        float3 _5302;
        _5302 = float3(0.0);
        _5301 = 0.0;
        float _5588;
        float3 _5589;
        for (int _5300 = 0; _5300 < 4; _5302 = _5589, _5301 = _5588, _5300++)
        {
            int2 _3189 = _3260 + _5710[_5300];
            int _3312 = _3189.x;
            bool _3315 = _3312 >= cbFI.interpolationRectBase.x;
            bool _3326;
            if (_3315)
            {
                _3326 = _3312 < (cbFI.interpolationRectSize.x + cbFI.interpolationRectBase.x);
            }
            else
            {
                _3326 = _3315;
            }
            bool _3334;
            if (_3326)
            {
                _3334 = _3189.y >= cbFI.interpolationRectBase.y;
            }
            else
            {
                _3334 = _3326;
            }
            bool _3345;
            if (_3334)
            {
                _3345 = _3189.y < (cbFI.interpolationRectSize.y + cbFI.interpolationRectBase.y);
            }
            else
            {
                _3345 = _3334;
            }
            if (_3345)
            {
                _5589 = _5302 + (r_current_interpolation_source.read(uint2(_3189), 0).xyz * _4658[_5300]);
                _5588 = _5301 + _4658[_5300];
            }
            else
            {
                _5589 = _5302;
                _5588 = _5301;
            }
        }
        float3 _5303;
        if (_5301 != 0.0)
        {
            _5303 = _5302 / float3(_5301);
        }
        else
        {
            _5303 = float3(0.0);
        }
        float2 _3670 = ((_1760 + (_5242 * _1772)) * _1759) - float2(0.5);
        int2 _3673 = int2(floor(_3670));
        float2 _3713 = fract(_3670);
        float _3682 = _3713.x;
        float _3683 = 1.0 - _3682;
        float _3685 = _3713.y;
        float _3686 = 1.0 - _3685;
        spvUnsafeArray<float, 4> _4948 = spvUnsafeArray<float, 4>({ _3683 * _3686, _3682 * _3686, _3683 * _3685, _3682 * _3685 });
        spvUnsafeArray<float, 4> _4618 = _4948;
        float _5337;
        float3 _5338;
        _5338 = float3(0.0);
        _5337 = 0.0;
        float _5606;
        float3 _5607;
        for (int _5336 = 0; _5336 < 4; _5338 = _5607, _5337 = _5606, _5336++)
        {
            int2 _3602 = _3673 + _5710[_5336];
            int _3725 = _3602.x;
            bool _3728 = _3725 >= cbFI.interpolationRectBase.x;
            bool _3739;
            if (_3728)
            {
                _3739 = _3725 < (cbFI.interpolationRectSize.x + cbFI.interpolationRectBase.x);
            }
            else
            {
                _3739 = _3728;
            }
            bool _3747;
            if (_3739)
            {
                _3747 = _3602.y >= cbFI.interpolationRectBase.y;
            }
            else
            {
                _3747 = _3739;
            }
            bool _3758;
            if (_3747)
            {
                _3758 = _3602.y < (cbFI.interpolationRectSize.y + cbFI.interpolationRectBase.y);
            }
            else
            {
                _3758 = _3747;
            }
            if (_3758)
            {
                _5607 = _5338 + (r_previous_interpolation_source.read(uint2(_3602), 0).xyz * _4618[_5336]);
                _5606 = _5337 + _4618[_5336];
            }
            else
            {
                _5607 = _5338;
                _5606 = _5337;
            }
        }
        float3 _5339;
        if (_5337 != 0.0)
        {
            _5339 = _5338 / float3(_5337);
        }
        else
        {
            _5339 = float3(0.0);
        }
        float2 _4083 = ((_1760 + ((-_5242) * _1772)) * _1759) - float2(0.5);
        int2 _4086 = int2(floor(_4083));
        float2 _4126 = fract(_4083);
        float _4095 = _4126.x;
        float _4096 = 1.0 - _4095;
        float _4098 = _4126.y;
        float _4099 = 1.0 - _4098;
        spvUnsafeArray<float, 4> _4930 = spvUnsafeArray<float, 4>({ _4096 * _4099, _4095 * _4099, _4096 * _4098, _4095 * _4098 });
        spvUnsafeArray<float, 4> _4578 = _4930;
        float _5376;
        float3 _5377;
        _5377 = float3(0.0);
        _5376 = 0.0;
        float _5629;
        float3 _5630;
        for (int _5375 = 0; _5375 < 4; _5377 = _5630, _5376 = _5629, _5375++)
        {
            int2 _4015 = _4086 + _5710[_5375];
            int _4138 = _4015.x;
            bool _4141 = _4138 >= cbFI.interpolationRectBase.x;
            bool _4152;
            if (_4141)
            {
                _4152 = _4138 < (cbFI.interpolationRectSize.x + cbFI.interpolationRectBase.x);
            }
            else
            {
                _4152 = _4141;
            }
            bool _4160;
            if (_4152)
            {
                _4160 = _4015.y >= cbFI.interpolationRectBase.y;
            }
            else
            {
                _4160 = _4152;
            }
            bool _4171;
            if (_4160)
            {
                _4171 = _4015.y < (cbFI.interpolationRectSize.y + cbFI.interpolationRectBase.y);
            }
            else
            {
                _4171 = _4160;
            }
            if (_4171)
            {
                _5630 = _5377 + (r_current_interpolation_source.read(uint2(_4015), 0).xyz * _4578[_5375]);
                _5629 = _5376 + _4578[_5375];
            }
            else
            {
                _5630 = _5377;
                _5629 = _5376;
            }
        }
        float3 _5378;
        if (_5376 != 0.0)
        {
            _5378 = _5377 / float3(_5376);
        }
        else
        {
            _5378 = float3(0.0);
        }
        float _1821 = _1784.x * float(_2329);
        float _1829 = _1784.y * float(_2306);
        bool _1866 = _5267 == 0.0;
        float3 _5470;
        if (_1866)
        {
            _5470 = _5303;
        }
        else
        {
            _5470 = select(mix(_5269, _5303, float3(fast::clamp((0.5 + (0.5 * (1.0 - _1821))) - (0.5 * (1.0 - _1829)), 0.0, 1.0))), _5269, bool3(_5301 == 0.0));
        }
        bool _1887;
        if (_1866)
        {
            _1887 = _5301 == 0.0;
        }
        else
        {
            _1887 = _1866;
        }
        bool _1892 = _5337 > 0.0;
        bool _1898;
        if (_1892)
        {
            _1898 = _5376 > 0.0;
        }
        else
        {
            _1898 = _1892;
        }
        float _5460;
        if (_1898)
        {
            _5460 = 0.5;
        }
        else
        {
            _5460 = _1892 ? 0.0 : 1.0;
        }
        float _4456 = fast::max(length(_5339), length(_5378));
        float _5462;
        if (_4456 > 0.0)
        {
            float3 _4443 = float3(_4456);
            _5462 = dot(_5339 / _4443, _5378 / _4443);
        }
        else
        {
            _5462 = 1.0;
        }
        float _4483 = fast::max(length(_5269), length(_5303));
        float _5463;
        if (_4483 > 0.0)
        {
            float3 _4470 = float3(_4483);
            _5463 = dot(_5269 / _4470, _5303 / _4470);
        }
        else
        {
            _5463 = 1.0;
        }
        _5570 = _1887 ? 1.0 : fast::clamp(fast::max(0.0, float(length(float2(_1821, _1829)) <= 0.001000000047497451305389404296875)), 0.0, 1.0);
        _5569 = mix(mix(_5339, _5378, float3(_5460)), _5470, float3(fast::clamp(mix(powr(fast::clamp(mix(fast::max(0.001000000047497451305389404296875, _5463), 1.0, fast::clamp(fast::clamp(1.0 - fast::min(_1821, _1829), 0.0, 1.0), 0.0, 1.0)) / fast::max(0.001000000047497451305389404296875, _5462), 0.0, 1.0), 1.0), 1.0, float(r_counters.data[1] < 10u)), 0.0, 1.0)));
    }
    rw_output.write(float4(_5569, _5570), uint2(_1558));
}


}

// ===== AMD FSR reconstruct_and_dilate =====
namespace ism_reconstruct_and_dilate {
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wmissing-braces"


using namespace metal;

template<typename T, size_t Num>
struct spvUnsafeArray
{
    T elements[Num ? Num : 1];
    
    thread T& operator [] (size_t pos) thread
    {
        return elements[pos];
    }
    constexpr const thread T& operator [] (size_t pos) const thread
    {
        return elements[pos];
    }
    
    device T& operator [] (size_t pos) device
    {
        return elements[pos];
    }
    constexpr const device T& operator [] (size_t pos) const device
    {
        return elements[pos];
    }
    
    constexpr const constant T& operator [] (size_t pos) const constant
    {
        return elements[pos];
    }
    
    threadgroup T& operator [] (size_t pos) threadgroup
    {
        return elements[pos];
    }
    constexpr const threadgroup T& operator [] (size_t pos) const threadgroup
    {
        return elements[pos];
    }
};

struct cbFI_t
{
    int2 renderSize;
    int2 displaySize;
    float2 displaySizeRcp;
    float cameraNear;
    float cameraFar;
    int2 upscalerTargetSize;
    int Mode;
    int reset;
    float4 fDeviceToViewDepth;
    float deltaTime;
    int HUDLessAttachedFactor;
    int2 distortionFieldSize;
    float2 opticalFlowScale;
    int opticalFlowBlockSize;
    uint dispatchFlags;
    int2 maxRenderSize;
    int opticalFlowHalfResMode;
    int NumInstances;
    int2 interpolationRectBase;
    int2 interpolationRectSize;
    packed_float3 debugBarColor;
    uint backBufferTransferFunction;
    float2 minMaxLuminance;
    float fTanHalfFOV;
    int _pad1;
    float2 fJitter;
    float2 fMotionVectorScale;
};

constant uint3 gl_WorkGroupSize [[maybe_unused]] = uint3(8u, 8u, 1u);

constant spvUnsafeArray<int2, 9> _361 = spvUnsafeArray<int2, 9>({ int2(0), int2(1, 0), int2(0, 1), int2(0, -1), int2(-1, 0), int2(-1, 1), int2(1), int2(-1), int2(1, -1) });
constant spvUnsafeArray<int2, 4> _922 = spvUnsafeArray<int2, 4>({ int2(0), int2(1, 0), int2(0, 1), int2(1) });

[[host_name("ism_fsr_reconstruct_and_dilate")]] kernel void ism_fsr_reconstruct_and_dilate(constant cbFI_t& cbFI [[buffer(0)]], texture2d<float> r_input_depth [[texture(0)]], texture2d<float> r_input_motion_vectors [[texture(1)]], texture2d<float, access::write> rw_dilated_motion_vectors [[texture(2)]], texture2d<float, access::write> rw_dilated_depth [[texture(3)]], texture2d<uint, access::read_write> rw_reconstructed_depth_previous_frame [[texture(4)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    int2 _462 = int2(gl_GlobalInvocationID.xy);
    spvUnsafeArray<float, 9> _533;
    for (int _907 = 0; _907 < 9; )
    {
        _533[_907] = r_input_depth.read(uint2((_462 + _361[_907])), 0).x;
        _907++;
        continue;
    }
    float _909;
    int2 _910;
    _910 = _462;
    _909 = _533[0];
    float _916;
    int2 _918;
    for (int _908 = 1; _908 < 9; _910 = _918, _909 = _916, _908++)
    {
        int2 _571 = _462 + _361[_908];
        if (all(uint2(_571) < uint2(cbFI.renderSize)))
        {
            bool _581 = _533[_908] < _909;
            _918 = select(_910, _571, bool2(_581));
            _916 = _581 ? _533[_908] : _909;
        }
        else
        {
            _918 = _910;
            _916 = _909;
        }
    }
    float2 _653 = float2(cbFI.renderSize);
    float2 _656 = float2(cbFI.displaySize);
    float2 _684 = r_input_motion_vectors.read(uint2(int2(floor((((float2(_910) + float2(0.5)) - cbFI.fJitter) / _653) * _656))), 0).xy * cbFI.fMotionVectorScale;
    rw_dilated_depth.write(float4(_909, 0.0, 0.0, 0.0), uint2(_462));
    rw_dilated_motion_vectors.write(float4(_684, 0.0, 0.0), uint2(_462));
    float2 _791 = ((((float2(_462) + float2(0.5)) / _653) + (_684 * float(length(_684 * _656) > 0.100000001490116119384765625))) * _653) - float2(0.5);
    int2 _794 = int2(floor(_791));
    float2 _834 = fract(_791);
    float _803 = _834.x;
    float _804 = 1.0 - _803;
    float _806 = _834.y;
    float _807 = 1.0 - _806;
    spvUnsafeArray<float, 4> _898 = spvUnsafeArray<float, 4>({ _804 * _807, _803 * _807, _804 * _806, _803 * _806 });
    spvUnsafeArray<float, 4> _874 = _898;
    for (int _911 = 0; _911 < 4; _911++)
    {
        if (_874[_911] > 0.001000000047497451305389404296875)
        {
            int2 _759 = _794 + _922[_911];
            if (all(uint2(_759) < uint2(cbFI.renderSize)))
            {
                uint _851 = rw_reconstructed_depth_previous_frame.atomic_fetch_min(uint2(_759), as_type<uint>(_909)).x;
            }
        }
    }
}


}

// ===== AMD FSR reconstruct_previous_depth =====
namespace ism_reconstruct_previous_depth {
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wmissing-braces"


using namespace metal;

template<typename T, size_t Num>
struct spvUnsafeArray
{
    T elements[Num ? Num : 1];
    
    thread T& operator [] (size_t pos) thread
    {
        return elements[pos];
    }
    constexpr const thread T& operator [] (size_t pos) const thread
    {
        return elements[pos];
    }
    
    device T& operator [] (size_t pos) device
    {
        return elements[pos];
    }
    constexpr const device T& operator [] (size_t pos) const device
    {
        return elements[pos];
    }
    
    constexpr const constant T& operator [] (size_t pos) const constant
    {
        return elements[pos];
    }
    
    threadgroup T& operator [] (size_t pos) threadgroup
    {
        return elements[pos];
    }
    constexpr const threadgroup T& operator [] (size_t pos) const threadgroup
    {
        return elements[pos];
    }
};

struct cbFI_t
{
    int2 renderSize;
    int2 displaySize;
    float2 displaySizeRcp;
    float cameraNear;
    float cameraFar;
    int2 upscalerTargetSize;
    int Mode;
    int reset;
    float4 fDeviceToViewDepth;
    float deltaTime;
    int HUDLessAttachedFactor;
    int2 distortionFieldSize;
    float2 opticalFlowScale;
    int opticalFlowBlockSize;
    uint dispatchFlags;
    int2 maxRenderSize;
    int opticalFlowHalfResMode;
    int NumInstances;
    int2 interpolationRectBase;
    int2 interpolationRectSize;
    packed_float3 debugBarColor;
    uint backBufferTransferFunction;
    float2 minMaxLuminance;
    float fTanHalfFOV;
    int _pad1;
    float2 fJitter;
    float2 fMotionVectorScale;
};

constant uint3 gl_WorkGroupSize [[maybe_unused]] = uint3(8u, 8u, 1u);

constant spvUnsafeArray<int2, 4> _603 = spvUnsafeArray<int2, 4>({ int2(0), int2(1, 0), int2(0, 1), int2(1) });

[[host_name("ism_fsr_reconstruct_previous_depth")]] kernel void ism_fsr_reconstruct_previous_depth(constant cbFI_t& cbFI [[buffer(0)]], texture2d<float> r_dilated_motion_vectors [[texture(0)]], texture2d<float> r_dilated_depth [[texture(1)]], texture2d<float> r_input_distortion_field [[texture(2)]], texture2d<uint, access::read_write> rw_reconstructed_depth_interpolated_frame [[texture(3)]], sampler s_LinearClamp [[sampler(0)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    int2 _311 = int2(gl_GlobalInvocationID.xy);
    float2 _349 = float2(cbFI.renderSize);
    float2 _350 = (float2(_311) + float2(0.5)) / _349;
    int2 _360 = _311 + int2(r_input_distortion_field.sample(s_LinearClamp, _350, level(0.0)).xy * _349);
    float4 _398 = r_dilated_depth.read(uint2(_360), 0);
    float _399 = _398.x;
    float2 _478 = ((_350 + (r_dilated_motion_vectors.read(uint2(_360), 0).xy * 0.5)) * _349) - float2(0.5);
    int2 _481 = int2(floor(_478));
    float2 _521 = fract(_478);
    float _490 = _521.x;
    float _491 = 1.0 - _490;
    float _493 = _521.y;
    float _494 = 1.0 - _493;
    spvUnsafeArray<float, 4> _590 = spvUnsafeArray<float, 4>({ _491 * _494, _490 * _494, _491 * _493, _490 * _493 });
    spvUnsafeArray<float, 4> _566 = _590;
    for (int _599 = 0; _599 < 4; _599++)
    {
        int2 _440 = _481 + _603[_599];
        if (_566[_599] > 0.001000000047497451305389404296875)
        {
            if (all(uint2(_440) < uint2(cbFI.renderSize)))
            {
                uint _543 = rw_reconstructed_depth_interpolated_frame.atomic_fetch_min(uint2(_440), as_type<uint>(_399)).x;
            }
        }
    }
}


}

// ===== AMD FSR setup =====
namespace ism_setup {

using namespace metal;

struct cbFI_t
{
    int2 renderSize;
    int2 displaySize;
    float2 displaySizeRcp;
    float cameraNear;
    float cameraFar;
    int2 upscalerTargetSize;
    int Mode;
    int reset;
    float4 fDeviceToViewDepth;
    float deltaTime;
    int HUDLessAttachedFactor;
    int2 distortionFieldSize;
    float2 opticalFlowScale;
    int opticalFlowBlockSize;
    uint dispatchFlags;
    int2 maxRenderSize;
    int opticalFlowHalfResMode;
    int NumInstances;
    int2 interpolationRectBase;
    int2 interpolationRectSize;
    packed_float3 debugBarColor;
    uint backBufferTransferFunction;
    float2 minMaxLuminance;
    float fTanHalfFOV;
    int _pad1;
    float2 fJitter;
    float2 fMotionVectorScale;
};

struct FrameInterpolationRWCounters_t
{
    uint data[1];
};

constant uint3 gl_WorkGroupSize [[maybe_unused]] = uint3(8u, 8u, 1u);

[[host_name("ism_fsr_setup")]] kernel void ism_fsr_setup(constant cbFI_t& cbFI [[buffer(0)]], coherent device FrameInterpolationRWCounters_t& rw_counters [[buffer(1)]], texture2d<uint> r_optical_flow_scd [[texture(0)]], texture2d<float, access::write> rw_disocclusion_mask [[texture(1)]], texture2d<uint, access::write> rw_game_motion_vector_field_x [[texture(2)]], texture2d<uint, access::write> rw_game_motion_vector_field_y [[texture(3)]], texture2d<uint, access::write> rw_optical_flow_motion_vector_field_x [[texture(4)]], texture2d<uint, access::write> rw_optical_flow_motion_vector_field_y [[texture(5)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    int2 _187 = int2(gl_GlobalInvocationID.xy);
    rw_counters.data[0] = 0u;
    if (all(_187 == int2(0)))
    {
        bool _267 = cbFI.reset == 1;
        bool _238;
        if (!_267)
        {
            _238 = (r_optical_flow_scd.read(uint2(int2(1, 0)), 0).x & 15u) != 0u;
        }
        else
        {
            _238 = _267;
        }
        if (_238)
        {
            rw_counters.data[1] = 0u;
        }
        else
        {
            rw_counters.data[1]++;
        }
    }
    rw_game_motion_vector_field_x.write(uint4(0u), uint2(_187));
    rw_game_motion_vector_field_y.write(uint4(0u), uint2(_187));
    rw_optical_flow_motion_vector_field_x.write(uint4(0u), uint2(_187));
    rw_optical_flow_motion_vector_field_y.write(uint4(0u), uint2(_187));
    rw_disocclusion_mask.write(float4(0.0), uint2(_187));
}


}
