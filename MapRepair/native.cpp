// Portable CPU implementation of the FSR1 FP32 EASU/RCAS equations used by ISM.
// FSR algorithms copyright (c) 2021 AMD. MIT notice: FSR_LICENSE.txt.
// The BVH triangle ray tracer and C interface below are MapRepair code.
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif
namespace {
struct V {float x,y,z; float& operator[](int i){return i==0?x:i==1?y:z;} float operator[](int i)const{return i==0?x:i==1?y:z;}};
V operator+(V a,V b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
V operator-(V a,V b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
V operator*(V a,float b){return {a.x*b,a.y*b,a.z*b};}
V operator*(V a,V b){return {a.x*b.x,a.y*b.y,a.z*b.z};}
V operator/(V a,float b){return a*(1.f/b);}
float dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}
V cross(V a,V b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
V norm(V a){return a/std::sqrt(std::max(dot(a,a),1e-30f));}
V lo(V a,V b){return {std::min(a.x,b.x),std::min(a.y,b.y),std::min(a.z,b.z)};}
V hi(V a,V b){return {std::max(a.x,b.x),std::max(a.y,b.y),std::max(a.z,b.z)};}
V clamp(V a,V l,V h){return lo(hi(a,l),h);}
float sat(float x){return std::clamp(x,0.f,1.f);}
float lum(V c){return .5f*c.x+c.y+.5f*c.z;}
struct P {V c;float a;};
P load(const uint8_t*s,int x,int y,int w,int h){
    x=std::clamp(x,0,w-1);y=std::clamp(y,0,h-1);const auto*p=s+(size_t(y)*w+x)*4;
    return {{p[0]/255.f,p[1]/255.f,p[2]/255.f},p[3]/255.f};
}
void store(uint8_t*d,P p){for(int c=0;c<3;c++)d[c]=uint8_t(std::nearbyint(sat(p.c[c])*255.f));d[3]=uint8_t(std::nearbyint(sat(p.a)*255.f));}
template<class F>void rows(int begin,int end,int threads,F fn){
    threads=std::clamp(threads,1,std::min(64,end-begin));std::atomic<int>next{begin};
    auto worker=[&]{for(int y;(y=next.fetch_add(1))<end;)fn(y);};
    std::vector<std::thread>pool;for(int i=1;i<threads;i++)pool.emplace_back(worker);
    worker();for(auto&t:pool)t.join();
}
P easu(const uint8_t*s,int w,int h,int x,int y){
    float xx=(x+.5f)*.5f-.5f,yy=(y+.5f)*.5f-.5f;int bx=int(std::floor(xx)),by=int(std::floor(yy));
    float px=xx-bx,py=yy-by;
    constexpr int ox[12]={0,1,-1,0,0,-1,1,2,2,1,1,0},oy[12]={-1,-1,1,1,0,0,1,1,0,0,2,2};
    P t[12];float l[12],alpha=1;
    for(int k=0;k<12;k++){t[k]=load(s,bx+ox[k],by+oy[k],w,h);l[k]=lum(t[k].c);alpha=std::min(alpha,t[k].a);}
    if(alpha<1){
        float weight[4]={(1-px)*(1-py),px*(1-py),(1-px)*py,px*py};int ix[4]={4,9,3,6};
        V c{};float sum=0;for(int k=0;k<4;k++){float q=weight[k]*t[ix[k]].a;c=c+t[ix[k]].c*q;sum+=q;}
        float a=load(s,int(std::floor(xx+.5f)),int(std::floor(yy+.5f)),w,h).a;
        return {a>0?c/std::max(sum,1e-8f):V{},a};
    }
    float dx=0,dy=0,len=0;
    auto set=[&](float weight,float a,float b,float c,float d,float e){
        float gx=d-b,gy=e-a,mx=std::max(std::abs(d-c),std::abs(c-b)),my=std::max(std::abs(e-c),std::abs(c-a));
        float ex=sat(std::abs(gx)/std::max(mx,1e-8f)),ey=sat(std::abs(gy)/std::max(my,1e-8f));
        dx+=gx*weight;dy+=gy*weight;len+=(ex*ex+ey*ey)*weight;
    };
    set((1-px)*(1-py),l[0],l[5],l[4],l[9],l[3]);set(px*(1-py),l[1],l[4],l[9],l[8],l[6]);
    set((1-px)*py,l[4],l[2],l[3],l[6],l[11]);set(px*py,l[9],l[3],l[6],l[7],l[10]);
    float d2=dx*dx+dy*dy;if(d2<1.f/32768.f){dx=1;dy=0;}else{float d=std::sqrt(d2);dx/=d;dy/=d;}
    len*=.5f;len*=len;float stretch=(dx*dx+dy*dy)/std::max(std::abs(dx),std::abs(dy));
    float lx=1+(stretch-1)*len,ly=1-.5f*len,lobe=.5f+(.21f-.5f)*len,clip=1/lobe,total=0;V color{};
    for(int k=0;k<12;k++){
        float ax=ox[k]-px,ay=oy[k]-py,vx=(ax*dx+ay*dy)*lx,vy=(-ax*dy+ay*dx)*ly;
        float d=std::min(vx*vx+vy*vy,clip),wb=.4f*d-1,wa=lobe*d-1;
        wb=1.5625f*wb*wb-.5625f;float weight=wb*wa*wa;color=color+t[k].c*weight;total+=weight;
    }
    return {clamp(color/total,lo(lo(t[4].c,t[9].c),lo(t[3].c,t[6].c)),hi(hi(t[4].c,t[9].c),hi(t[3].c,t[6].c))),1};
}
P rcas(const uint8_t*s,int w,int h,int x,int y,float strength){
    P b=load(s,x,y-1,w,h),d=load(s,x-1,y,w,h),e=load(s,x,y,w,h),f=load(s,x+1,y,w,h),v=load(s,x,y+1,w,h);
    if(std::min({b.a,d.a,e.a,f.a,v.a})<1)return e;
    float bl=lum(b.c),dl=lum(d.c),el=lum(e.c),fl=lum(f.c),vl=lum(v.c);
    float range=std::max({bl,dl,el,fl,vl})-std::min({bl,dl,el,fl,vl});
    float noise=1-.5f*sat(std::abs(.25f*(bl+dl+fl+vl)-el)/std::max(range,1e-8f));
    V mn=lo(lo(b.c,d.c),lo(f.c,v.c)),mx=hi(hi(b.c,d.c),hi(f.c,v.c));float lobe=-1;
    for(int c=0;c<3;c++){
        float hitMin=std::min(mn[c],e.c[c])/std::max(4*mx[c],1e-8f);
        float hitMax=(1-std::max(mx[c],e.c[c]))/std::min(4*mn[c]-4,-1e-8f);
        lobe=std::max(lobe,std::max(-hitMin,hitMax));
    }
    lobe=std::clamp(lobe,-.1875f,0.f)*strength*noise;
    return {((b.c+d.c+v.c+f.c)*lobe+e.c)/(4*lobe+1),e.a};
}
thread_local char error[512];
struct Tri{V a,b,c,normal,min,max,center;};
struct Node{V min,max;int begin,end,left=-1,right=-1;};
struct Scene{
    std::vector<Tri>tris;std::vector<int>ids;std::vector<Node>nodes;
    Scene(const float*v,int n){
        for(int i=0;i<n;i++){V a{v[9*i],v[9*i+1],v[9*i+2]},b{v[9*i+3],v[9*i+4],v[9*i+5]},c{v[9*i+6],v[9*i+7],v[9*i+8]};
            tris.push_back({a,b,c,norm(cross(b-a,c-a)),lo(a,lo(b,c)),hi(a,hi(b,c)),(a+b+c)/3.f});}
        ids.resize(n);std::iota(ids.begin(),ids.end(),0);build(0,n);
    }
    int build(int b,int e){
        V l{INFINITY,INFINITY,INFINITY},h{-INFINITY,-INFINITY,-INFINITY};
        for(int k=b;k<e;k++){l=lo(l,tris[ids[k]].min);h=hi(h,tris[ids[k]].max);}
        int index=int(nodes.size());nodes.push_back({l,h,b,e});
        if(e-b>8){V d=h-l;int axis=d.y>d.x?1:0;if(d.z>d[axis])axis=2;int mid=(b+e)/2;
            std::nth_element(ids.begin()+b,ids.begin()+mid,ids.begin()+e,[&](int i,int j){return tris[i].center[axis]<tris[j].center[axis];});
            int left=build(b,mid),right=build(mid,e);nodes[index].left=left;nodes[index].right=right;}
        return index;
    }
    bool box(const Node&n,V o,V d,float limit)const{
        float near=0,far=limit;for(int c=0;c<3;c++){
            if(std::abs(d[c])<1e-12f){if(o[c]<n.min[c]||o[c]>n.max[c])return false;continue;}
            float a=(n.min[c]-o[c])/d[c],b=(n.max[c]-o[c])/d[c];if(a>b)std::swap(a,b);
            near=std::max(near,a);far=std::min(far,b);if(near>far)return false;}
        return true;
    }
    int hit(V o,V d,float&dist)const{
        int stack[128],size=1,result=-1;stack[0]=0;
        while(size){const auto&n=nodes[stack[--size]];if(!box(n,o,d,dist))continue;
            if(n.left>=0){stack[size++]=n.left;stack[size++]=n.right;continue;}
            for(int k=n.begin;k<n.end;k++){int id=ids[k];const auto&t=tris[id];V e1=t.b-t.a,e2=t.c-t.a,p=cross(d,e2);float det=dot(e1,p);
                if(std::abs(det)<1e-12f)continue;float inv=1/det;V s=o-t.a;float u=dot(s,p)*inv;if(u<0||u>1)continue;
                V q=cross(s,e1);float v=dot(d,q)*inv;if(v<0||u+v>1)continue;float z=dot(e2,q)*inv;
                if(z>.0001f&&z<dist){dist=z;result=id;}}
        }return result;
    }
};
V basis(V n,V local){V a=norm(cross(std::abs(n.z)<.99f?V{0,0,1}:V{0,1,0},n));return a*local.x+cross(n,a)*local.y+n*local.z;}
}
API const char*mr_error(){return error;}
API int mr_fsr_rows(const uint8_t*src,int w,int h,uint8_t*dst,int first,int count,int pass,float sharp,int threads){
    try{
        int ow=pass?w:w*2,oh=pass?h:h*2;
        if(!src||!dst||w<1||h<1||w>100000||h>100000||first<0||count<1||first+count>oh||!std::isfinite(sharp)||sharp<0||sharp>1)throw std::runtime_error("Invalid FSR dimensions/strength");
        rows(first,first+count,threads,[&](int y){for(int x=0;x<ow;x++)store(dst+(size_t(y-first)*ow+x)*4,pass?rcas(src,w,h,x,y,sharp):easu(src,w,h,x,y));});return 0;
    }catch(const std::exception&e){std::snprintf(error,sizeof(error),"%s",e.what());return -1;}
}
// Orthographic primary rays, direct sun visibility, cosine hemisphere sky and
// up to two diffuse bounces. Mesh XY coordinates are map pixels; Z uses the same scale.
API int mr_trace(const float*vertices,int n,float*output,int gw,int gh,float extentx,float extenty,float camera_z,float elevation,float azimuth,int samples,int threads){
    try{
        if(!vertices||!output||n<1||n>1000000||gw<1||gh<1||gw>512||gh>512||samples<1||samples>64)throw std::runtime_error("Invalid ray scene");
        Scene scene(vertices,n);const float pi=3.141592653589793f;float e=elevation*pi/180,a=azimuth*pi/180;
        V sun=norm({std::sin(a)*std::cos(e),-std::cos(a)*std::cos(e),std::sin(e)}),sky{.97f,1.f,1.04f},sunColor{1.08f,1.f,.87f};
        V reference=sky*.75f+sunColor*.25f;
        rows(0,gh,threads,[&](int y){for(int x=0;x<gw;x++){
            V camera{(x+.5f)/gw*extentx-.5f,(y+.5f)/gh*extenty-.5f,camera_z},direction{0,0,-1};float dist=INFINITY;
            int id=scene.hit(camera,direction,dist);V total{};
            if(id<0)total=reference*float(samples);
            else for(int s=0;s<samples;s++){
                V origin=camera+direction*dist,normv=scene.tris[id].normal;if(dot(normv,direction)>0)normv=normv*-1;
                V throughput{1,1,1},radiance{};
                for(int bounce=0;bounce<2;bounce++){
                    float u=(s+.5f)/samples,v=std::fmod((s+.5f)*.61803398875f+bounce*.38196601125f,1.f),phi=2*pi*v;
                    V o=origin+normv*.0002f;float shadow=INFINITY;
                    if(dot(normv,sun)>0&&scene.hit(o,sun,shadow)<0)radiance=radiance+throughput*sunColor*(.25f*std::max(dot(normv,sun),0.f)/std::max(sun.z,.001f));
                    V ray=basis(normv,{std::sqrt(u)*std::cos(phi),std::sqrt(u)*std::sin(phi),std::sqrt(1-u)});float t=INFINITY;int hit=scene.hit(o,ray,t);
                    if(hit<0){radiance=radiance+throughput*sky*.75f;break;}
                    origin=o+ray*t;normv=scene.tris[hit].normal;if(dot(normv,ray)>0)normv=normv*-1;throughput=throughput*.5f;
                }total=total+radiance;
            }
            size_t k=(size_t(y)*gw+x)*3;for(int c=0;c<3;c++)output[k+c]=total[c]/samples/reference[c];
        }});return 0;
    }catch(const std::exception&e){std::snprintf(error,sizeof(error),"%s",e.what());return -1;}
}
