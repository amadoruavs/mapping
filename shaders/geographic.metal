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
