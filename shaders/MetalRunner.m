#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <simd/simd.h>

static int fail(NSString *message) { fprintf(stderr,"%s\n",message.UTF8String); return 1; }

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 10) return fail(@"Usage: MetalRunner shader rgb normals visibility output elevation azimuth strength count");
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return fail(@"No Metal GPU is available to this process");
    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
    if (!source) return fail(error.description);
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    if (!library) return fail(error.description);
    id<MTLFunction> function = [library newFunctionWithName:@"golden_hour"];
    if (!function) return fail(@"Missing golden_hour kernel");
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) return fail(error.description);
    uint32_t count = (uint32_t)strtoul(argv[9],NULL,10);
    if (!count) return fail(@"Pixel count must be positive");
    id<MTLBuffer> inputs[3];
    for (int i=0; i<3; i++) {
      NSData *data = [NSData dataWithContentsOfFile:@(argv[i+2]) options:0 error:&error];
      NSUInteger size = (NSUInteger)count*(i==2 ? 4 : 16);
      if (!data || data.length != size) return fail(@"Input buffer length mismatch");
      inputs[i] = [device newBufferWithBytes:data.bytes length:size options:MTLResourceStorageModeShared];
      if (!inputs[i]) return fail(@"GPU input allocation failed");
    }
    id<MTLBuffer> output = [device newBufferWithLength:(NSUInteger)count*16 options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!output || !encoder) return fail(@"GPU output allocation failed");
    [encoder setComputePipelineState:pipeline];
    for (int i=0;i<3;i++) [encoder setBuffer:inputs[i] offset:0 atIndex:i];
    [encoder setBuffer:output offset:0 atIndex:3];
    vector_float4 parameters = {strtof(argv[6],NULL),strtof(argv[7],NULL),strtof(argv[8],NULL),0};
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:4];
    [encoder setBytes:&count length:sizeof(count) atIndex:5];
    NSUInteger threads = MIN(pipeline.maxTotalThreadsPerThreadgroup,256);
    [encoder dispatchThreadgroups:MTLSizeMake((count+threads-1)/threads,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
    if (command.error) return fail(command.error.description);
    NSData *result = [NSData dataWithBytes:output.contents length:(NSUInteger)count*16];
    if (![result writeToFile:@(argv[5]) options:NSDataWritingAtomic error:&error]) return fail(error.description);
    printf("Metal device: %s, pixels: %u\n",device.name.UTF8String,count);
  }
  return 0;
}
