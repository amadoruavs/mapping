#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

// Small synchronous C ABI; Python owns buffer lifetimes. Encodes are asynchronous
// until adm_sync, so a sequence of MSL and MPS operations shares one submission.
typedef struct { uint32_t u[16]; float f[32]; } ADMParams;
@interface ADMContext : NSObject
@property id<MTLDevice> device;
@property id<MTLCommandQueue> queue;
@property id<MTLLibrary> library;
@property id<MTLCommandBuffer> command;
@property NSMutableDictionary *pipelines;
@property NSMutableDictionary *filters;
@property NSMutableDictionary *products;
@property NSMutableDictionary *scratch;
@property NSString *error;
@property double gpuSeconds;
@property uint64_t submissions;
@property uint64_t dispatches;
@property uint64_t mpsOperations;
@end
@implementation ADMContext
@end
static NSString *creationError;
static ADMContext *ctx(void *p) { return (__bridge ADMContext *)p; }
static id<MTLBuffer> buf(void *p) { return (__bridge id<MTLBuffer>)p; }
static int failure(ADMContext *c, NSString *s) { c.error=s; return -1; }
static id<MTLCommandBuffer> command(ADMContext *c) {
    if (!c.command) { c.command=[c.queue commandBuffer]; c.command.label=@"ADM MSL + MPS batch"; }
    return c.command;
}
static id<MTLComputePipelineState> pipeline(ADMContext *c, NSString *name) {
    id<MTLComputePipelineState> p=c.pipelines[name];
    if (!p) {
        NSError *error=nil;
        id<MTLFunction> f=[c.library newFunctionWithName:name];
        if (!f) { failure(c,[@"Missing kernel: " stringByAppendingString:name]); return nil; }
        p=[c.device newComputePipelineStateWithFunction:f error:&error];
        if (!p) { failure(c,error.description); return nil; }
        c.pipelines[name]=p;
    }
    return p;
}
void *adm_create(const char *path) {
    @autoreleasepool {
        ADMContext *c=[ADMContext new]; c.device=MTLCreateSystemDefaultDevice();
        if (!c.device || !MPSSupportsMTLDevice(c.device)) { creationError=@"No MPS-capable Metal device is visible to this process"; return NULL; }
        NSError *error=nil;
        NSString *source=[NSString stringWithContentsOfFile:@(path) encoding:NSUTF8StringEncoding error:&error];
        if (!source) { creationError=error.description; return NULL; }
        for (NSString *extra in @[@"alignment.metal", @"coherent.metal", @"lasso.metal"]) {
            NSString *file=[[@(path) stringByDeletingLastPathComponent] stringByAppendingPathComponent:extra];
            if ([[NSFileManager defaultManager] fileExistsAtPath:file]) {
                NSString *part=[NSString stringWithContentsOfFile:file encoding:NSUTF8StringEncoding error:&error];
                if (!part) { creationError=error.description; return NULL; }
                source=[source stringByAppendingFormat:@"\n%@\n",part];
            }
        }
        MTLCompileOptions *options=[MTLCompileOptions new];
        // Preserve cancellation-sensitive NCC and photometric threshold arithmetic.
        if (@available(macOS 15.0, *)) options.mathMode=MTLMathModeSafe;
        else options.fastMathEnabled=NO;
        c.library=[c.device newLibraryWithSource:source options:options error:&error];
        if (!c.library) { creationError=error.description; return NULL; }
        c.queue=[c.device newCommandQueue]; c.pipelines=[NSMutableDictionary new];
        c.filters=[NSMutableDictionary new]; c.products=[NSMutableDictionary new]; c.scratch=[NSMutableDictionary new];
        return (__bridge_retained void *)c;
    }
}
// Every exported function drains its own pool: Python calls in on a thread with no
// autorelease pool, so any autoreleased retain would otherwise leak the object.
const char *adm_error(void *p) { @autoreleasepool { return (p ? ctx(p).error : creationError).UTF8String; } }
const char *adm_device(void *p) { @autoreleasepool { return ctx(p).device.name.UTF8String; } }
uint64_t adm_max_buffer(void *p) { @autoreleasepool { return ctx(p).device.maxBufferLength; } }
void adm_release(void *p) { @autoreleasepool { if (p) { id object=CFBridgingRelease(p); (void)object; } } }
uint64_t adm_allocated(void *p) { @autoreleasepool { return ctx(p).device.currentAllocatedSize; } }
void *adm_buffer(void *p, uint64_t size) {
    @autoreleasepool {
        ADMContext *c=ctx(p);
        if (!size || size>c.device.maxBufferLength) { failure(c,@"Buffer exceeds Metal allocation limit"); return NULL; }
        id<MTLBuffer> b=[c.device newBufferWithLength:size options:MTLResourceStorageModeShared];
        if (!b) { failure(c,@"Metal shared buffer allocation failed"); return NULL; }
        return (__bridge_retained void *)b;
    }
}
void *adm_contents(void *p) { @autoreleasepool { return buf(p).contents; } }
int adm_sync(void *p) {
    @autoreleasepool {
        ADMContext *c=ctx(p); if (!c.command) return 0;
        id<MTLCommandBuffer> b=c.command; c.command=nil;
        [b commit]; [b waitUntilCompleted]; c.submissions++;
        c.gpuSeconds+=MAX(0,b.GPUEndTime-b.GPUStartTime);
        if (b.status!=MTLCommandBufferStatusCompleted) return failure(c,b.error.description ?: @"Metal command failed");
        return 0;
    }
}
void adm_stats(void *p, double *v) {
    @autoreleasepool {
        ADMContext *c=ctx(p); v[0]=c.gpuSeconds; v[1]=c.submissions;
        v[2]=c.dispatches; v[3]=c.mpsOperations;
    }
}
int adm_dispatch(void *p, const char *name, void **buffers, uint32_t n,
                 const ADMParams *parameters, uint32_t threads, uint32_t groupSize) {
    @autoreleasepool {
        ADMContext *c=ctx(p); id<MTLComputePipelineState> ps=pipeline(c,@(name));
        if (!ps || n>15 || !threads) return failure(c,c.error ?: @"Invalid dispatch");
        NSUInteger group=groupSize ?: MIN(256,ps.maxTotalThreadsPerThreadgroup);
        if (group>ps.maxTotalThreadsPerThreadgroup) return failure(c,@"Threadgroup exceeds kernel limit");
        if ((strcmp(name,"ncc")==0 || strcmp(name,"nearest_two")==0) && ps.threadExecutionWidth!=32)
            return failure(c,@"Cooperative matching kernels require a 32-lane SIMD group");
        id<MTLComputeCommandEncoder> e=[command(c) computeCommandEncoder];
        e.label=@(name); [e setComputePipelineState:ps];
        for (uint32_t i=0;i<n;i++) [e setBuffer:buf(buffers[i]) offset:0 atIndex:i];
        [e setBytes:parameters length:sizeof(*parameters) atIndex:15];
        [e dispatchThreadgroups:MTLSizeMake((threads+group-1)/group,1,1) threadsPerThreadgroup:MTLSizeMake(group,1,1)];
        [e endEncoding]; c.dispatches++; return 0;
    }
}
int adm_matmul(void *p, void *a, void *b, void *out,
               uint32_t rows, uint32_t cols, uint32_t inner, uint64_t offset) {
    @autoreleasepool {
        ADMContext *c=ctx(p);
        // A[rows,inner] B[cols,inner]^T; descriptor row strides are explicit.
        MPSMatrixDescriptor *da=[MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:inner rowBytes:inner*4 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *db=[MPSMatrixDescriptor matrixDescriptorWithRows:cols columns:inner rowBytes:inner*4 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *dc=[MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols rowBytes:cols*4 dataType:MPSDataTypeFloat32];
        MPSMatrix *ma=[[MPSMatrix alloc] initWithBuffer:buf(a) offset:offset descriptor:da];
        MPSMatrix *mb=[[MPSMatrix alloc] initWithBuffer:buf(b) descriptor:db];
        MPSMatrix *mc=[[MPSMatrix alloc] initWithBuffer:buf(out) descriptor:dc];
        NSString *key=[NSString stringWithFormat:@"%u/%u/%u",rows,cols,inner];
        MPSMatrixMultiplication *op=c.products[key];
        if (!op) {
            op=[[MPSMatrixMultiplication alloc] initWithDevice:c.device transposeLeft:NO transposeRight:YES resultRows:rows resultColumns:cols interiorColumns:inner alpha:1 beta:0];
            // Every image pair has its own shape; bound the kernel cache in pair count.
            if (c.products.count>=64) [c.products removeAllObjects];
            c.products[key]=op;
        }
        [op encodeToCommandBuffer:command(c) leftMatrix:ma rightMatrix:mb resultMatrix:mc];
        c.mpsOperations++; return 0;
    }
}
int adm_gaussian(void *p, void *input, void *output, uint32_t width, uint32_t height,
                 const float *weights, uint32_t size) {
    @autoreleasepool {
        ADMContext *c=ctx(p);
        if (!width || !height || !(size&1)) return failure(c,@"Invalid Gaussian dimensions");
        NSData *key=[NSData dataWithBytes:weights length:size*4];
        NSArray *ops=c.filters[key];
        if (!ops) {
            MPSImageConvolution *x=[[MPSImageConvolution alloc] initWithDevice:c.device kernelWidth:size kernelHeight:1 weights:weights];
            MPSImageConvolution *y=[[MPSImageConvolution alloc] initWithDevice:c.device kernelWidth:1 kernelHeight:size weights:weights];
            x.edgeMode=MPSImageEdgeModeClamp; y.edgeMode=MPSImageEdgeModeClamp;
            ops=@[x,y]; c.filters[key]=ops;
        }
        uint32_t radius=size/2, tw=MIN(width,2048u), th=MIN(height,256u);
        MTLTextureDescriptor *d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:tw+2*radius height:th+2*radius mipmapped:NO];
        d.storageMode=MTLStorageModePrivate; d.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;
        NSString *scratchKey=[NSString stringWithFormat:@"%lu/%lu",(unsigned long)d.width,(unsigned long)d.height];
        NSArray *textures=c.scratch[scratchKey];
        if (!textures) {
            id<MTLTexture> a=[c.device newTextureWithDescriptor:d], b=[c.device newTextureWithDescriptor:d], z=[c.device newTextureWithDescriptor:d];
            if (!a || !b || !z) return failure(c,@"MPS scratch texture allocation failed");
            textures=@[a,b,z];
            // Bound cached scratch across different map shapes.
            if (c.scratch.count>=4) [c.scratch removeAllObjects];
            c.scratch[scratchKey]=textures;
        }
        id<MTLTexture> a=textures[0], b=textures[1], z=textures[2];
        if (!a || !b || !z) return failure(c,@"MPS scratch texture allocation failed");
        id<MTLComputePipelineState> pad=pipeline(c,@"pad_texture"), crop=pipeline(c,@"crop_texture");
        if (!pad || !crop) return -1;
        // Reuse three bounded textures for every tile; tracked hazards order reuse.
        for (uint32_t y=0;y<height;y+=th) for (uint32_t x=0;x<width;x+=tw) {
            ADMParams q={0}; uint32_t v[]={width,height,x,y,tw,th,radius,tw+2*radius,th+2*radius};
            memcpy(q.u,v,sizeof(v));
            id<MTLComputeCommandEncoder> e=[command(c) computeCommandEncoder];
            [e setComputePipelineState:pad]; [e setBuffer:buf(input) offset:0 atIndex:0]; [e setTexture:a atIndex:0];
            [e setBytes:&q length:sizeof(q) atIndex:15];
            [e dispatchThreads:MTLSizeMake(d.width,d.height,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)]; [e endEncoding];
            [ops[0] encodeToCommandBuffer:command(c) sourceTexture:a destinationTexture:b];
            [ops[1] encodeToCommandBuffer:command(c) sourceTexture:b destinationTexture:z];
            e=[command(c) computeCommandEncoder]; [e setComputePipelineState:crop];
            [e setBuffer:buf(output) offset:0 atIndex:0]; [e setTexture:z atIndex:0];
            [e setBytes:&q length:sizeof(q) atIndex:15];
            [e dispatchThreads:MTLSizeMake(tw,th,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)]; [e endEncoding];
            c.dispatches+=2; c.mpsOperations+=2;
        }
        return 0;
    }
}

// One host operation. All derivatives, normal equations, Cholesky solves,
// damping decisions and parameter updates stay on the GPU. Check only the
// convergence flag between four-iteration batches to avoid unused solves.
int adm_align(void *p, void **inputs, uint32_t n, uint32_t edges,
              uint32_t observations, uint32_t anchor, uint32_t iterations, float width) {
    @autoreleasepool {
        ADMContext *c=ctx(p);NSUInteger d=n*8;
        if (!n||!edges||!observations||anchor>=n||!iterations||width<=0) return failure(c,@"Invalid alignment dimensions");
        NSMutableArray *keep=[NSMutableArray array];
        id<MTLBuffer> (^allocate)(NSUInteger)=^id<MTLBuffer>(NSUInteger bytes){
            id<MTLBuffer> b=[c.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if(b)[keep addObject:b];return b;
        };
        id<MTLBuffer> x=buf(inputs[0]),points=buf(inputs[1]),owners=buf(inputs[2]),edge=buf(inputs[3]);
        id<MTLBuffer> links=buf(inputs[4]),offsets=buf(inputs[5]),incidence=buf(inputs[6]),state=buf(inputs[7]);
        id<MTLBuffer> jac=allocate(observations*32*4),rw=allocate(observations*16),blocks=allocate(edges*272*4);
        id<MTLBuffer> normal=allocate(d*d*4),factor=allocate(d*d*4),rhs=allocate(d*4),delta=allocate(d*4),trial=allocate(d*4),status=allocate(4);
        id<MTLBuffer> cost=allocate(observations*4),trialCost=allocate(observations*4),ones=allocate(observations*4),oldSum=allocate(4),newSum=allocate(4);
        if(!jac||!rw||!blocks||!normal||!factor||!rhs||!delta||!trial||!status||!cost||!trialCost||!ones||!oldSum||!newSum)return failure(c,@"Alignment allocation failed");
        float *values=ones.contents;for(uint32_t i=0;i<observations;i++)values[i]=1.f;
        MPSMatrixDescriptor *square=[MPSMatrixDescriptor matrixDescriptorWithRows:d columns:d rowBytes:d*4 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *vector=[MPSMatrixDescriptor matrixDescriptorWithRows:d columns:1 rowBytes:4 dataType:MPSDataTypeFloat32];
        MPSMatrix *A=[[MPSMatrix alloc] initWithBuffer:normal descriptor:square],*L=[[MPSMatrix alloc] initWithBuffer:factor descriptor:square];
        MPSMatrix *b=[[MPSMatrix alloc] initWithBuffer:rhs descriptor:vector],*solution=[[MPSMatrix alloc] initWithBuffer:delta descriptor:vector];
        MPSMatrixDecompositionCholesky *chol=[[MPSMatrixDecompositionCholesky alloc] initWithDevice:c.device lower:YES order:d];
        MPSMatrixSolveCholesky *solve=[[MPSMatrixSolveCholesky alloc] initWithDevice:c.device upper:NO order:d numberOfRightHandSides:1];
        int (^dispatch)(const char *,NSArray *,NSUInteger,ADMParams)=^int(const char *name,NSArray *buffers,NSUInteger count,ADMParams q){
            void *handles[15];for(NSUInteger i=0;i<buffers.count;i++)handles[i]=(__bridge void *)buffers[i];
            return adm_dispatch(p,name,handles,(uint32_t)buffers.count,&q,(uint32_t)count,0);
        };
        ADMParams q={0};q.u[0]=observations;q.f[0]=width;
        for(uint32_t iteration=0;iteration<iterations;iteration++){
            if(dispatch("align_linearize",@[x,points,owners,jac,rw,cost],observations,q))return -1;
            ADMParams e={0};e.u[0]=edges;
            if(dispatch("align_edge_blocks",@[jac,rw,edge,blocks],edges*272,e))return -1;
            ADMParams a={0};a.u[0]=n;a.u[1]=anchor;
            if(dispatch("align_assemble",@[blocks,edge,links,offsets,incidence,normal,rhs,state],d*d+d,a))return -1;
            [chol encodeToCommandBuffer:command(c) sourceMatrix:A resultMatrix:L status:status];
            [solve encodeToCommandBuffer:command(c) sourceMatrix:L rightHandSideMatrix:b solutionMatrix:solution];c.mpsOperations+=2;
            ADMParams v={0};v.u[0]=(uint32_t)d;
            if(dispatch("align_trial",@[x,delta,trial,status,state],d,v))return -1;
            if(dispatch("align_cost",@[trial,points,owners,trialCost],observations,q))return -1;
            if(adm_matmul(p,(__bridge void *)ones,(__bridge void *)cost,(__bridge void *)oldSum,1,1,observations,0))return -1;
            if(adm_matmul(p,(__bridge void *)ones,(__bridge void *)trialCost,(__bridge void *)newSum,1,1,observations,0))return -1;
            ADMParams it={0};it.u[0]=iteration;it.u[1]=(uint32_t)d;
            if(dispatch("align_accept",@[oldSum,newSum,status,state,delta,x],1,it))return -1;
            if(dispatch("align_commit",@[x,trial,state],d,v))return -1;
            if((iteration+1)%4==0 || iteration+1==iterations){
                if(adm_sync(p))return -1;
                float *diagnostics=state.contents;
                diagnostics[9]=iteration+1;
                if(diagnostics[6])break;
            }
        }
        return adm_sync(p);
    }
}
