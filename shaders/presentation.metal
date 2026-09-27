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
