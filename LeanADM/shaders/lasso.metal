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
    for(int j=-18;j<=18;j++) {
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
