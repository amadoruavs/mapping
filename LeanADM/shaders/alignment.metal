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
kernel void align_accept(device const float *oldCost [[buffer(0)]],device const float *newCost [[buffer(1)]],device const int *status [[buffer(2)]],device float *state [[buffer(3)]],device const float *delta [[buffer(4)]],device const float *x [[buffer(5)]],constant Params &p [[buffer(15)]],uint k [[thread_position_in_grid]]) {
    if(k)return;state[1]=0;if(state[6])return;
    float old=oldCost[0],next=newCost[0];if(p.u[0]==0)state[7]=old;
    if(status[0]==0&&isfinite(next)&&next<old){
        state[1]=1;state[2]++;state[0]=max(state[0]*.35f,1e-7f);state[4]=next;
        if(old-next<1e-6f*max(old,1.f))state[6]=1;
    }else{state[3]++;state[0]=min(state[0]*8.f,1e12f);state[4]=old;}
    float step=0,norm=0;for(uint i=0;i<p.u[1];i++){step+=delta[i]*delta[i];norm+=x[i]*x[i];}
    uint trace=16+p.u[0]*4;state[trace]=old;state[trace+1]=next;state[trace+2]=state[0];state[trace+3]=sqrt(step/max(norm,1.f));
    // About 2.5 float32 ulps: smaller normalized updates oscillate in the
    // projection arithmetic. Low damping is required to exclude artificial stalls.
    if(status[0]==0&&(state[2]>0||old<=1e-8f)&&state[0]<.01f&&step<=9e-14f*max(norm,1.f))state[6]=2;
    state[5]=p.u[0]+1;state[8]=status[0];
}
kernel void align_commit(device float *x [[buffer(0)]],device const float *trial [[buffer(1)]],device const float *state [[buffer(2)]],constant Params &p [[buffer(15)]],uint k [[thread_position_in_grid]]) {
    if(k<p.u[0]&&state[1])x[k]=trial[k];
}
