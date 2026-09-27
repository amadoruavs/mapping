// Thin adapter around Project-10/OpenRT, BSD-3-Clause. Upstream is unmodified.
// Trace neutral diffuse geometry under sun + deterministic sky samples, then
// normalize by the identical unoccluded map plane to retain photographed light.
#include "openrt.h"
#include "core/Ray.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

class FixedSky final : public rt::CSampler {
public:
    FixedSky():CSampler(4){}
    Vec2f getNextSample(size_t i) const override {
        return Vec2f((float(i%4)+.5f)/4, (float(i/4)+.5f)/4);
    }
};

void add_mesh(rt::CScene& scene,const std::vector<float>& data,size_t count,float unit){
    auto shader=std::make_shared<rt::CShaderDiffuse>(scene,Vec3f::all(1),0);
    for(size_t i=0;i<count;i++){
        const float* v=data.data()+9*i;
        Vec3f a(v[0]/unit,v[1]/unit,v[2]/unit),b(v[3]/unit,v[4]/unit,v[5]/unit),c(v[6]/unit,v[7]/unit,v[8]/unit);
        if(norm((b-a).cross(c-a))<1e-10)continue;
        auto p=scene.addTriangle(a,b,c);p->setShader(shader);
    }
    scene.buildAccelStructure();
}

void add_lights(rt::CScene& scene,float elevation,float azimuth,float aspect){
    const float e=elevation*Pif/180,a=azimuth*Pif/180,dist=1000;
    const Vec3f sun(std::sin(a)*std::cos(e),-std::cos(a)*std::cos(e),std::sin(e));
    scene.add(std::make_shared<rt::CLightOmni>(Vec3f::all(.4f*Pif*dist*dist/std::sin(e)),
        Vec3f(.5f,.5f*aspect,0)+dist*sun));
    scene.add(std::make_shared<rt::CLightSky>(Vec3f::all(.6f*Pif),20,std::make_shared<FixedSky>()));
}

int main(int argc,char** argv){
    try{
        if(argc!=11)throw std::runtime_error("Expected: mesh.bin field.bin map_width map_height grid_width grid_height elevation azimuth threads triangle_count");
        int w=std::stoi(argv[3]),h=std::stoi(argv[4]),gw=std::stoi(argv[5]),gh=std::stoi(argv[6]);
        float elevation=std::stof(argv[7]),azimuth=std::stof(argv[8]);int threads=std::stoi(argv[9]);
        size_t count=std::stoull(argv[10]);
        if(w<1||h<1||gw<1||gh<1||gw>2048||gh>2048||count<2||count>1000000||threads<1||threads>256||
           !std::isfinite(elevation)||elevation<1||elevation>90||!std::isfinite(azimuth))throw std::runtime_error("Invalid trace settings");
        std::vector<float> vertices(count*9);std::ifstream input(argv[1],std::ios::binary);
        if(!input.read(reinterpret_cast<char*>(vertices.data()),vertices.size()*sizeof(float)))throw std::runtime_error("Incomplete triangle input");
        for(float v:vertices)if(!std::isfinite(v))throw std::runtime_error("Nonfinite geometry");
        const float unit=float(std::max(w,h));float height=2;
        for(size_t i=2;i<vertices.size();i+=3)height=std::max(height,vertices[i]/unit+1);
        rt::CScene scene,reference;
        add_mesh(scene,vertices,count,unit);add_mesh(reference,vertices,2,unit);
        add_lights(scene,elevation,azimuth,float(h)/w);add_lights(reference,elevation,azimuth,float(h)/w);
        std::vector<float> field(size_t(gw)*gh*3,1);cv::setNumThreads(threads);
        cv::parallel_for_(cv::Range(0,gh),[&](const cv::Range& range){
            for(int y=range.start;y<range.end;y++)for(int x=0;x<gw;x++){
                Vec3f origin(((x+.5f)*w/gw-.5f)/unit,((y+.5f)*h/gh-.5f)/unit,height);
                rt::Ray ray(origin,Vec3f(0,0,-1)),base(origin,Vec3f(0,0,-1));
                Vec3f color=scene.rayTrace(ray),neutral=reference.rayTrace(base);
                for(int c=0;c<3;c++)field[(size_t(y)*gw+x)*3+c]=color[c]/std::max(neutral[c],1e-6f);
            }
        });
        for(float v:field)if(!std::isfinite(v))throw std::runtime_error("Nonfinite OpenRT lighting");
        std::ofstream output(argv[2],std::ios::binary);
        if(!output.write(reinterpret_cast<char*>(field.data()),field.size()*sizeof(float)))throw std::runtime_error("Could not write lighting field");
        std::cout<<"OpenRT: "<<gw<<"x"<<gh<<", "<<count<<" triangles, sun + 16 sky samples, plane-normalized\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
