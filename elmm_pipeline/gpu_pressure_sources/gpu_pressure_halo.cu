#include "scoped_runtime.h"
#include <cuda_runtime.h>
#include <cstdint>
struct PressureShape { int32_t n[3],bc[6]; };
__global__ void pressure_faces(double *phi,PressureShape shape,size_t elements) {
    for(size_t flat=size_t(blockIdx.x)*blockDim.x+threadIdx.x;flat<elements;flat+=size_t(gridDim.x)*blockDim.x) {
        const size_t nx=size_t(shape.n[0])+4,ny=size_t(shape.n[1])+4;
        size_t q[3]={flat%nx,(flat/nx)%ny,flat/(nx*ny)};int outside=0,axis=-1;
        for(int i=0;i<3;++i) if(q[i]<2 || q[i]>=size_t(shape.n[i])+2) { ++outside;axis=i; }
        if(outside!=1) continue; // Original Bound_Phi leaves edges/corners alone.
        const bool upper=q[axis]>=size_t(shape.n[axis])+2;
        if(shape.bc[2*axis+int(upper)]==0) q[axis]=upper?q[axis]-shape.n[axis]:q[axis]+shape.n[axis];
        else q[axis]=upper?2*size_t(shape.n[axis])+3-q[axis]:3-q[axis];
        phi[flat]=phi[q[0]+nx*(q[1]+ny*q[2])];
    }
}
extern "C" int elmm_gpu_pressure_launch_halo_v1(double *phi,const int32_t *sizes,const int32_t *bcs,void *stream,int *started) {
    PressureShape shape{};size_t elements=1;
    for(int i=0;i<3;++i) { shape.n[i]=sizes[i];elements*=size_t(sizes[i])+4; }
    for(int i=0;i<6;++i) shape.bc[i]=bcs[i];
    const size_t blocks=(elements+255)/256;
    *started=1;
    pressure_faces<<<unsigned(blocks<65535?blocks:65535),256,0,static_cast<cudaStream_t>(stream)>>>(phi,shape,elements);
    return cudaGetLastError()==cudaSuccess?FORT_SCOPE_OK:FORT_SCOPE_EXECUTION;
}
