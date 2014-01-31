
OpenCL is a framework for writing programs that can be run specialized compute devices like GPU.

The way OpenCL works is like this :
Programmer writes a small program written OpenCL C, usually in a .cl file
This program contains at least 1 function that has the attribute 'kernel'

A sofware can then read the source of the OpenCL C and give it to OpenCL.
OpenCL then gives the source to the device driver which compiles it for the desired device.
Afterward, the software asks OpenCL to send data to the device memory.
Then the software asks OpenCL to run the kernel with the data.
After that, the software asks for the result of the computation to be transfered back to the host (to the CPU).

Some terminology :
device         The device used by OpenCL to run the kernels, usually a GPU

host           The environment using OpenCL, usually a software running on a CPU

kernel         A function marked with 'kernel' in OpenCL C
               Must be return void
               Usually takes at least one pointer as argument
               Compiled by the driver
               Executed in parralel on the device

program        Code written in OpenCL C

OpenCL C       Language based on C (version C99 + additions)

Global range   Total number of work items to start for one kernel execution
               (number of time the kernel will be executed)

work-item      One instance of execution of a kernel, also called a 'thread' or a 'worker'

work-group     Work items are launched in groups.
               Each group is completely independant from other groups (no synchronization, no data sharing)
               A common work-group size 256
               The maximum work-group size is dependent on the device used
               work-items in the same workgroup can be synchonized
               
Local range    Number of work items per work-group

NDRange        A list of numbers representing the size of ranges (Global range and Local range)
               ND is for 'N' dimentions, meaning the 'problem size' can have multiple dimentions
               The NDRange is usually made of 3 values, so 3 dimentions
               The size of the range is each of the value multiplied together
               For image processing, we use 2 dimentions, which give a range similar to this :
                  Global : 512, 512, 1  - Which gives a total of 262 144 work items
                  Local  : 16, 16, 1    - Which gives 256 work-items per workgroup

buffer         A segment of memory on the device
               Viewed as a sequence of bytes by the hardware of the device

image          An image in the memory of the device (1D, 2D or 3D)
               Viewed as a 'texture' by the hardware of the device
               The device knows the type of data and number of channels in the image

global         A type of device memory that all work-items can read and write
               This is the slowest type of memory in the device

constant       A type of device memory that work-items can only read from
               Similar to global but faster because it is usually cached
               A limited amount of constant memory is available (sizes depend on the device used)

local          A type of device memory that is local to one work-group
               Each work-group has one instance of its local memory
               Work-items in one work-group can read and write in their work-group's local memory
               local memory is much faster than global memory
               local memory can be used to share data between items in the same workgroup
               A very limited amount of constant memory is available (sizes depend on the device used, can be as low as 16KB)

private        A type of device memory that is independent for each work-item
               It is the fastest type of memory
               All variables declared without 'global, constant or local' will be private
               Only a very limited amount is available (usually a few KB)

read_only      An image that is given as input to a kernel
               Can only be read
               Faster than global memory and usually as fast as constant memory
               Uses texture hardware on GPUs
               
write_only     An image that is given as output to a kernel
               Can only be written
               
sampler        An objects that tells the device how to read from a read_only image
               Can set :
                  What to return when reading outside of image
                  How to interpret in-between pixel coordinates

barrier        A point in a kernel that all work-items in a work-group must reach
                  before any work-item in the work-group can continue.
               Used to synchronize data exchanges between work-items inside a work-group

platform       Environment provided a the device vendor that implements OpenCL
               Usual platforms are : AMD, nVidia, Intel, etc...

context        Object representing a collection of devices from the same platform

Command queue  Object representing a list of actions to do on one device of a context
               These actions are mainly kernel executions and memory transfers
               Actions are asyncronous, they will be done when the device(s) is available
               The command queue can either be set 'in-order' (the default) or 'out-of-order'
               When 'in-order', each action is started after the previous action is done
               When 'out-of-order', multiple actions can be executed at the same time and
                  not necessarily in the order they were issued
               'out-of-order' can give more performance trough increased device utilisation,
                  at the cost of more complex programming

event          A syncronization object used to order execution of dependent actions in 'out-of-order' command queues



C API for OpenCL

OpenCL is provided with a C API
The API is Object Oriented
Because we use C++ and C++ handled Object Oriented programming better than C, we won't use the C API directly


C++ bindings for OpenCL

There exist an official C++ bindings for the OpenCL API, which is well made and reduces the amount of work needed to make a C++ software that uses OpenCL.


How to make a working OpenCL C++ program :

#include <cl/cl.hpp>
#include <vector>

using namespace cl;
using namespace std;

int main(int, char**)
{
   // Create a platform
   Platform platform = Platform::getDefault();
   
   // List devices for this platform
   vector<Device> devices;
   platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
   
   // Create a context for the first device
   Context context(devices[0]);
   
   // Create a command queue
   CommandQueue queue(context, devices[0]);
   
   // Create a program
   Program program(context, "OpenCL C code goes here...");
   
   // Build the program
   program.build();
   
   // Create an object representing our kernel
   auto kernel = make_kernel<Buffer, Buffer>(program, "example_kernel");
   
   // Prepare 
   const static int Size = 1000000;
   vector<int> inputData(Size, 0), outputData(Size, 0);
   
   // Create the buffers in the device
   Buffer inputBuffer(context, CL_MEM_READ_ONLY, Size * sizeof(int));      // Can only be read by the kernel
   Buffer outputBuffer(context, CL_MEM_WRITE_ONLY, Size * sizeof(int));    // Can only be written by the kernel
   
   // Send the inputData to the device memory
   queue.enqueueWriteBuffer(inputBuffer, false, 0, Size * sizeof(int), inputData.data());
   
   // Enqueue execution of the kernel
   //    With a Global size of 1000000 and an automatic local size
   //    The kernel will be executed 1000000 times
   kernel(EnqueueArgs(queue, NDRange(Size)), inputBuffer, outputBuffer);
   
   // Read outputData from the device memory
   queue.enqueueReadBuffer(outputBuffer, false, 0, Size * sizeof(int), outputData.data());
   
   // Wait for all queued actions to complete
   queue.finish();
   
   // outputData now contains the result of the computation
}



How to write an OpenCL C program :

// This program would work if given to the above example where "OpenCL C code goes here..."
kernel void example_kernel(global int * input, global int * output)
{
   int worker_id = get_global_id(0);   // Retreives the id of the current work-item
   // Each work item will have a different id, starting from 0 to the first value given to NDRange
   
   output[worker_id] = input[worker_id] + 10 + worker_id;   // Example computation
}

// OpenCL C has the C pre-processor :
#define MACRO(a, b) a + b

// More advanced
bool function(int a)
{
   float4 vector_type(0, 1, 2, 3);  // OpenCL C adds vector data types 
   // These vector data types simplify 3D calculations, pixel color manipulations and other similar tasks
   // They also ease the task of the compiler when trying to generate SIMD code
   
   vector_type *= 2; // They work just like normal basic types
   
   float v = vector_type.x;   // Individual members can be accessed with .x, .y, .z and .w  or  .S0, .S1, .S2 and.S3
   float2 v2; float8 v8; float16 v16   // Multiple sizes are available
   uchar uc; uint ui;   // unsigned types are simpler to write
   
   local bool local_buffer[256];    // in local memory - can't have an initializer
   int lid = get_local_id(0);       // worker no in the workgroup
   if (lid < 256)
      local_buffer[lid] = (uc8.S1 == uc);
      
   barrier(CLK_LOCAL_MEM_FENCE);    // synch work-items in the workgroup and issue all local memory operations
   
   if (lid < 256 && lid > 1)
      return local_buffer[lid - 1]; // Read result from another work item in the same work group
      
   return false;
}


Performance notes about OpenCL :
Program build time
   Programs can take a long time to build (100+ms), make sure to compile the program only once and
      keep the built program ready for when it is needed

Memory transfer
   Transferering memory from host to device and back takes time (1+ms), make sure to transfer only needed data and transfer them only when needed

Global data access
   Devices generally have more compute power than they have global memory bandwith
   kernels that read multiple values from global memory can be accelerated by copying the data in a local array

Amount of computation per kernel
   A very simple kernel that read data, do few calculations and then write the result do not use the full power of a GPU.
   A sequence of simple kernels will perform less than one kernel doing all the calculations at once.
   But a very big kernel can suffer from private or local memory exhaustion on some devices and will have less performance
      (this is usually not a problem except for very complex algorithms).


More reading :
C++ Wrapper for OpenCL : http://www.khronos.org/registry/cl/specs/opencl-cplusplus-1.2.pdf
OpenCL Reference : http://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/
Quick reference card : http://www.khronos.org/registry/cl/sdk/1.2/docs/OpenCL-1.2-refcard.pdf
Complete OpenCL tutorial : http://www.cmsoft.com.br/index.php?option=com_content&view=category&layout=blog&id=41&Itemid=75
   (uses .Net for host software but it still a very good and complete tutorial)
