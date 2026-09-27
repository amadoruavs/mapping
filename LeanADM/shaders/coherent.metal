// Low-resolution observation vectors drive coherent source labels. Full-size
// output pixels always come from exactly one photograph: no color-mean blending.
kernel void coherent_observation(device const packed_uchar3 *im [[buffer(0)]],device uchar4 *obs [[buffer(1)]],device half *confidence [[buffer(2)]],device float4 *mean [[buffer(3)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint w=p.u[0],h=p.u[1],gw=p.u[2],gh=p.u[3],count=gw*gh;if(i>=count)return;
    float2 xy=transform(float2(i%gw,i/gw),p);float2 d=min(xy,float2(w-1,h-1)-xy)/float2(w*.5f,h*.5f);float c=min(d.x,d.y);
    bool valid=all(isfinite(xy))&&c>=.02f;c=valid?c*c:0;float3 rgb=valid?interpolate(im,xy,w,h,p.u[15]):float3(0);
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
kernel void coherent_render(device const packed_uchar3 *im [[buffer(0)]],device const ushort *labels [[buffer(1)]],device float *best [[buffer(2)]],device uchar4 *selected [[buffer(3)]],device ushort *coverage [[buffer(4)]],device ushort *ids [[buffer(5)]],device const float4 *bias [[buffer(6)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint w=p.u[0],h=p.u[1],rw=p.u[2],rh=p.u[3];if(i>=rw*rh)return;
    uint x=i%rw+p.u[4],y=i/rw+p.u[5],j=y*p.u[6]+x;
    float2 xy=transform(float2(i%rw,i/rw),p);float2 d=min(xy,float2(w-1,h-1)-xy)/float2(w*.5f,h*.5f);float c=min(d.x,d.y);
    if(!all(isfinite(xy))||c<.02f)return;
    uint lx=min(uint((x+.5f)*p.u[8]/p.u[6]),p.u[8]-1),ly=min(uint((y+.5f)*p.u[9]/p.u[7]),p.u[9]-1);
    float score=c+(labels[ly*p.u[8]+lx]==p.u[10]?2.f:0.f);if(p.u[11])coverage[j]++;
    if(score>best[j]){
        float2 q=clamp((float2(x,y)+.5f)*float2(p.u[8],p.u[9])/float2(p.u[6],p.u[7])-.5f,float2(0),float2(p.u[8]-1,p.u[9]-1));
        uint2 b=uint2(q),next=min(b+1,uint2(p.u[8]-1,p.u[9]-1));float2 f=fract(q);
        float4 correction=mix(mix(bias[b.y*p.u[8]+b.x],bias[b.y*p.u[8]+next.x],f.x),mix(bias[next.y*p.u[8]+b.x],bias[next.y*p.u[8]+next.x],f.x),f.y);
        float3 rgb=interpolate(im,xy,w,h,p.u[15])+clamp(correction.xyz/max(correction.w,1e-6f),-32.f,32.f);
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
