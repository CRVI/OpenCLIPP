
OpenCL is framework that is very similar to nVidia CUDA

Similarities :
Able to run code on GPU
Based on an extended version of the C language
kernels are launched with a Grid size and a Block size (named Global range and Local range in OpenCL)

It has these major differences :
OpenCL is supported by multiple vendors : nVidia, AMD, Intel and many others
OpenCL C programs stay in source form until they are used (they are not built with compiling the host-side software)
In CUDA, kernels can be called directly by host code while in OpenCL, the kernel calling process is more labourious.
OpenCL C does not have templates


Terminology :
This lists terms used in CUDA and their equivalent in OpenCL

CUDA              OpenCL
Thread            Work-item
Thread block      Work-group
global memory     global memory
constant memory   constant memory
shared memory     local memory
local memory      private memory
Grid size         Global range
Block size        Local range
__global__        kernel
gridDim.x         get_num_groups(0)
blockDim.x        get_local_size(0)
blockIdx.x        get_group_id(0)
threadIds.x       get_local_id(0)
__syncthreads     barrier()
warp              no equivalent
