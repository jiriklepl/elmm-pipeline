// Explicit ELMM adapter: approved serial TOP/BOTTOM plane reference, followed
// by Bound_Pr face extrapolation. All kernels use the supplied runtime stream.
#include "scoped_runtime.h"
#include <cuda_runtime.h>
#include <array>
#include <cstdint>
#include <limits>

namespace {
struct Shape { int32_t n[3],bc[6],reference; double *scalar; };
__device__ size_t index(const size_t *q,const Shape &s) {
    return q[0]+size_t(s.n[0]+4)*(q[1]+size_t(s.n[1]+4)*q[2]);
}
__global__ void plane_reference(const double *p,Shape s) {
    double sum=0;
    if(s.reference) {
        const size_t k=s.reference==1?2:size_t(s.n[2])+1;
        // Preserve the approved Fortran j-then-i summation order exactly.
        for(int j=0;j<s.n[1];++j) for(int i=0;i<s.n[0];++i) {
            size_t q[3]={size_t(i)+2,size_t(j)+2,k};sum=sum+p[index(q,s)];
        }
        sum=sum/(s.n[0]*s.n[1]);
    }
    *s.scalar=sum;
}
__global__ void subtract_reference(double *p,Shape s,size_t count) {
    const double ref=*s.scalar;
    if(ref==0) return;
    for(size_t t=size_t(blockIdx.x)*blockDim.x+threadIdx.x;t<count;t+=size_t(gridDim.x)*blockDim.x) {
        size_t q[3]={t%size_t(s.n[0])+2,(t/size_t(s.n[0]))%size_t(s.n[1])+2,
                     t/(size_t(s.n[0])*s.n[1])+2};
        const size_t at=index(q,s);p[at]=p[at]-ref;
    }
}
__global__ void pressure_faces(double *p,Shape s,size_t count) {
    for(size_t t=size_t(blockIdx.x)*blockDim.x+threadIdx.x;t<count;t+=size_t(gridDim.x)*blockDim.x) {
        size_t offset=t;int axis=0;
        const size_t yz=size_t(s.n[1])*s.n[2],xz=size_t(s.n[0])*s.n[2];
        if(offset>=yz) { offset-=yz;axis=1;if(offset>=xz) { offset-=xz;axis=2; } }
        const int a=axis==0?1:0,b=axis==2?1:2;
        size_t q[3]={};q[a]=offset%size_t(s.n[a])+2;q[b]=offset/size_t(s.n[a])+2;
        const size_t pitch=axis==0?1:axis==1?size_t(s.n[0])+4:size_t(s.n[0]+4)*(s.n[1]+4);
        const size_t base=index(q,s),n=s.n[axis];
        if(s.bc[2*axis]==0) {
            p[base]=p[base+n*pitch];p[base+pitch]=p[base+(n+1)*pitch];
        } else {
            p[base+pitch]=p[base+4*pitch]-3*(p[base+3*pitch]-p[base+2*pitch]);
            p[base]=p[base+3*pitch];
        }
        if(s.bc[2*axis+1]==0) {
            p[base+(n+2)*pitch]=p[base+2*pitch];p[base+(n+3)*pitch]=p[base+3*pitch];
        } else {
            p[base+(n+2)*pitch]=p[base+(n-1)*pitch]-3*(p[base+n*pitch]-p[base+(n+1)*pitch]);
            p[base+(n+3)*pitch]=p[base+n*pitch];
        }
    }
}
int worker(void *user,const fort_scope_external_view_v1 *views,size_t count,void *stream,int *started) noexcept {
    *started=0;
    if(count!=1 || !views || !user) return FORT_SCOPE_ARGUMENT;
    const auto s=*static_cast<Shape *>(user);const auto &v=views[0];
    if(v.type!=FORT_SCOPE_REAL64 || v.rank!=3 || v.element_bytes!=8 || !v.device ||
       !v.extents || !v.strides || !v.lower_bounds) return FORT_SCOPE_ARGUMENT;
    size_t bytes=8,elements=1,faces=0;
    for(int axis=0;axis<3;++axis) {
        if(v.extents[axis]!=size_t(s.n[axis])+4 || v.strides[axis]!=bytes || v.lower_bounds[axis]!=-1)
            return FORT_SCOPE_ARGUMENT;
        bytes*=v.extents[axis];elements*=s.n[axis];
        faces+=size_t(s.n[(axis+1)%3])*s.n[(axis+2)%3];
    }
    if(v.bytes!=bytes) return FORT_SCOPE_ARGUMENT;
    auto *p=static_cast<double *>(v.device);auto cuda_stream=static_cast<cudaStream_t>(stream);
    *started=1;
    plane_reference<<<1,1,0,cuda_stream>>>(p,s);
    if(cudaGetLastError()!=cudaSuccess) return FORT_SCOPE_EXECUTION;
    const unsigned blocks=unsigned((elements+255)/256<65535?(elements+255)/256:65535);
    subtract_reference<<<blocks,256,0,cuda_stream>>>(p,s,elements);
    if(cudaGetLastError()!=cudaSuccess) return FORT_SCOPE_EXECUTION;
    const unsigned face_blocks=unsigned((faces+255)/256<65535?(faces+255)/256:65535);
    pressure_faces<<<face_blocks,256,0,cuda_stream>>>(p,s,faces);
    return cudaGetLastError()==cudaSuccess?FORT_SCOPE_OK:FORT_SCOPE_EXECUTION;
}
}

extern "C" int elmm_gpu_pressure_tail_v1(fort_scope_t context,fort_buffer_t pressure,
    const int32_t *sizes,const int32_t *bcs,int reference,fort_scope_external_report_v1 *report) noexcept {
    if(report) *report={1,FORT_SCOPE_ARGUMENT,0,0};
    if(!sizes || !bcs || !report || !pressure || reference<0 || reference>2) return FORT_SCOPE_ARGUMENT;
    Shape shape{};size_t elements=1;
    for(int a=0;a<3;++a) {
        if(sizes[a]<3 || sizes[a]>INT32_MAX-4 || elements>SIZE_MAX/size_t(sizes[a]+4)) return FORT_SCOPE_ARGUMENT;
        elements*=size_t(sizes[a]+4);shape.n[a]=sizes[a];
    }
    if(elements>SIZE_MAX/8 || int64_t(sizes[0])*sizes[1]>INT32_MAX) return FORT_SCOPE_ARGUMENT;
    for(int f=0;f<6;++f) { if(bcs[f]!=0 && bcs[f]!=4) return FORT_SCOPE_ARGUMENT;shape.bc[f]=bcs[f]; }
    shape.reference=reference;
    fort_scope_scratch_lease_v1 scratch{};
    int status=fort_scope_scratch_acquire_v1(context,sizeof(double),&scratch);
    if(status) { report->status=status;return status; }
    shape.scalar=static_cast<double *>(scratch.device);
    std::array<std::array<size_t,3>,7> lo,hi;
    lo[0]={2,2,2};for(int a=0;a<3;++a) hi[0][a]=size_t(sizes[a])+2;
    std::array<fort_scope_section,7> sections;
    sections[0]={lo[0].data(),hi[0].data()};
    for(int a=0;a<3;++a) for(int side=0;side<2;++side) {
        const int f=1+2*a+side;lo[f]=lo[0];hi[f]=hi[0];
        lo[f][a]=side?hi[0][a]:0;hi[f][a]=lo[f][a]+2;sections[f]={lo[f].data(),hi[f].data()};
    }
    // All interior values remain necessary; only the six faces are overwritten.
    // The transaction covers the serial reference and all dependent kernels.
    fort_scope_access access{0,1,sections.data(),7,sections.data(),6,sections.data()+1};
    fort_scope_external_binding_v1 binding{pressure,FORT_SCOPE_REAL64,3,8,access};
    fort_scope_external_call_v1 call{1,0,&binding,1,0,0};
    status=fort_scope_external_execute_v1(context,&call,worker,&shape,report);
    const int released=fort_scope_scratch_release_v1(context,scratch.token);
    if(!status) status=released;
    report->status=status;return status;
}
