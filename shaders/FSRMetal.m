// Resource/dispatch adapter for unmodified AMD GLSL translated to Metal.
// Shader licenses remain in third_party/fidelityfx_sdk.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@interface FSRContext : NSObject
@property id<MTLDevice> device;
@property id<MTLCommandQueue> queue;
@property id<MTLCommandBuffer> command;
@property NSMutableDictionary *pipelines;
@property NSArray *samplers;
@property NSString *error;
@end
@implementation FSRContext
@end
static NSString *creationError;
static FSRContext *context(void *p){return (__bridge FSRContext*)p;}
static id<MTLTexture> texture(void *p){return (__bridge id<MTLTexture>)p;}
static id<MTLBuffer> buffer(void *p){return (__bridge id<MTLBuffer>)p;}
static int fail(FSRContext *c,NSString *e){c.error=e;return -1;}
static id<MTLCommandBuffer> command(FSRContext *c){if(!c.command)c.command=[c.queue commandBuffer];return c.command;}
void fsr_release(void *p){@autoreleasepool{if(p)CFRelease(p);}}
const char *fsr_error(void *p){return (p?context(p).error:creationError).UTF8String;}
void *fsr_create(void){@autoreleasepool{
    FSRContext *c=[FSRContext new];c.device=MTLCreateSystemDefaultDevice();
    if(!c.device){creationError=@"No Metal GPU is visible";return NULL;}
    c.queue=[c.device newCommandQueue];c.pipelines=[NSMutableDictionary new];
    NSMutableArray *samplers=[NSMutableArray new];
    for(int linear=0;linear<2;linear++){
        MTLSamplerDescriptor *s=[MTLSamplerDescriptor new];s.minFilter=s.magFilter=linear?MTLSamplerMinMagFilterLinear:MTLSamplerMinMagFilterNearest;
        s.mipFilter=MTLSamplerMipFilterNearest;s.sAddressMode=s.tAddressMode=s.rAddressMode=MTLSamplerAddressModeClampToEdge;
        [samplers addObject:[c.device newSamplerStateWithDescriptor:s]];
    }c.samplers=samplers;return (__bridge_retained void*)c;
}}
int fsr_compile(void *p,const char *name,const char *source){@autoreleasepool{
    FSRContext *c=context(p);NSError *e=nil;MTLCompileOptions *o=[MTLCompileOptions new];
    if(@available(macOS 26.0,*))o.languageVersion=MTLLanguageVersion4_0;
    else return fail(c,@"FSR translated coherent shaders require macOS 26 / Metal 4");
    if(@available(macOS 15.0,*))o.mathMode=MTLMathModeSafe;else o.fastMathEnabled=NO;
    id<MTLLibrary> lib=[c.device newLibraryWithSource:[NSString stringWithUTF8String:source] options:o error:&e];
    if(!lib)return fail(c,e.description);
    id<MTLComputePipelineState> pipeline=[c.device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"main0"] error:&e];
    if(!pipeline)return fail(c,e.description);c.pipelines[@(name)]=pipeline;return 0;
}}
void *fsr_texture(void *p,uint32_t w,uint32_t h,uint32_t format,uint32_t levels){@autoreleasepool{
    FSRContext *c=context(p);MTLTextureDescriptor *d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:(MTLPixelFormat)format width:w height:h mipmapped:levels>1];
    d.mipmapLevelCount=levels;d.storageMode=MTLStorageModeShared;d.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite|MTLTextureUsagePixelFormatView;
    if(format==MTLPixelFormatR32Uint)d.usage|=MTLTextureUsageShaderAtomic;
    id<MTLTexture> t=[c.device newTextureWithDescriptor:d];if(!t){fail(c,@"FSR texture allocation failed");return NULL;}return (__bridge_retained void*)t;
}}
void *fsr_mip(void *p,uint32_t mip){@autoreleasepool{
    id<MTLTexture> t=texture(p);id<MTLTexture> v=[t newTextureViewWithPixelFormat:t.pixelFormat textureType:MTLTextureType2D levels:NSMakeRange(mip,1) slices:NSMakeRange(0,1)];
    return (__bridge_retained void*)v;
}}
void fsr_texture_write(void *p,const void *data,uint32_t rowBytes){@autoreleasepool{
    id<MTLTexture> t=texture(p);[t replaceRegion:MTLRegionMake2D(0,0,t.width,t.height) mipmapLevel:0 withBytes:data bytesPerRow:rowBytes];
}}
void fsr_texture_read(void *p,void *data,uint32_t rowBytes){@autoreleasepool{
    id<MTLTexture> t=texture(p);[t getBytes:data bytesPerRow:rowBytes fromRegion:MTLRegionMake2D(0,0,t.width,t.height) mipmapLevel:0];
}}
void *fsr_buffer(void *p,const void *data,uint32_t size){@autoreleasepool{
    id<MTLBuffer> b=data?[context(p).device newBufferWithBytes:data length:size options:MTLResourceStorageModeShared]:[context(p).device newBufferWithLength:size options:MTLResourceStorageModeShared];
    if(b&&!data)memset(b.contents,0,size);return (__bridge_retained void*)b;
}}
void fsr_buffer_write(void *p,const void *data,uint32_t size){memcpy(buffer(p).contents,data,size);}
int fsr_sync(void *p){@autoreleasepool{
    FSRContext *c=context(p);if(!c.command)return 0;id<MTLCommandBuffer> b=c.command;[b commit];[b waitUntilCompleted];c.command=nil;
    return b.status==MTLCommandBufferStatusError?fail(c,b.error.description):0;
}}
int fsr_dispatch(void *p,const char *name,void **textures,uint32_t nt,void **buffers,uint32_t nb,
    uint32_t *samplers,uint32_t ns,uint32_t *groups,uint32_t *threads){@autoreleasepool{
    FSRContext *c=context(p);id<MTLComputePipelineState> ps=c.pipelines[@(name)];if(!ps)return fail(c,@"Missing FSR pipeline");
    if((uint64_t)threads[0]*threads[1]*threads[2]>ps.maxTotalThreadsPerThreadgroup)return fail(c,@"FSR threadgroup exceeds device limit");
    id<MTLComputeCommandEncoder> e=[command(c) computeCommandEncoder];[e setComputePipelineState:ps];
    for(uint32_t i=0;i<nt;i++)if(textures[i])[e setTexture:texture(textures[i]) atIndex:i];
    for(uint32_t i=0;i<nb;i++)if(buffers[i])[e setBuffer:buffer(buffers[i]) offset:0 atIndex:i];
    for(uint32_t i=0;i<ns;i++)[e setSamplerState:c.samplers[samplers[i]] atIndex:i];
    [e dispatchThreadgroups:MTLSizeMake(groups[0],groups[1],groups[2]) threadsPerThreadgroup:MTLSizeMake(threads[0],threads[1],threads[2])];
    [e endEncoding];return 0;
}}
