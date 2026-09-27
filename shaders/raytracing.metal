#include <metal_raytracing>
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
