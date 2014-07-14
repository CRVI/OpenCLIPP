
The OpenCLIPP exposes a pure C interface that can be used by C programs or by any language that has C bindings.


Concepts

This library deals with image processing. To use the library it is necessary to understand these concepts :
1 Images
   OpenCL can represent images in device memory in two ways : As a Image or as a Buffer
   OpenCLIPP stores images in buffers.
   An Image must be created using a SImage structure to inform the library about the size and type of the image.
   When creating an Image, the memory space is allocated on the device
   That memory space will not be initialized.
   Use ocipSend* and ocipRead* to transfer data to and from the device.
2 Programs
   Before processing can be done, OpenCL needs to have a Program and needs to have that program built for the type of image used.
   Programs are in .cl files in a folder distributed with this library
   ocipSetCLFilesPath() needs to be called to tell the library where those files are
   ocipPrepare*() needs to be called to create the desired program and build it
3 Synchronous and asynchronous
   OpenCL works with an operation queue
   When an operation is issued, it is added to the queue
   All previous operations are guaranteed to be done before starting the next item in the queue.
   These operations can be memory transfers and processing functions
   Asynchronous functions :
      ocipSendImage
      Most processing functions
   These functions will return almost immediately.
   The image transfer or image processing will happens asyncrhonously.
   Read functions :
      ocipReadImage
   The read functions will issue a reading operation and then wait for all operations to complete.
   So after a read function returns, it is guaranteed to contain the resulting data of the previously issued operations.
4 Processing functions
   The processing functions are grouped by program
   To use a processing function, a handle to the corresponding program must be used.
   Each of these functions does the following :
      If the program was not built for the image type used, the program is built
      If the source images have not been sent to device memory, a Send operation is added to the queue for each source.
      The processing operation is added to the queue
   

Structures :

SImage   Contains information about an image


Types :

ocipError      Type returned by all ocip calls to signal errors
               No error is signaled by CL_SUCCESS
               
ocipContext    Context returned when ininitizing the library
               All calls to the library are related to the current context
               
ocipImage      Handle to an image on the device

ocipProgram    Handle to a program
               A handle to a program is needed to call most ocip functions
               

Functions :

ocipError ocip_API ocipInitialize(ocipContext * ContextPtr, const char * PreferredPlatform, cl_device_type deviceType);
Initializes OpenCL, creates an execution context, sets the new context as the current context
and returns the context handle.
The handle must be closed by calling ocipUninitialize when the context (or the whole library) is no longer needed.
ocipInitialize() can be called more than once, in that case, each context must be
released individually by a call to ocipUninitialize(). Images and Programs
created from different context can't be mixed (a program can only run
with images that have been created from the same context).

ocipError ocip_API ocipUninitialize(ocipContext Context)
Releases the context.

ocipError ocip_API ocipChangeContext(ocipContext Context);
Change the current context.
Advanced users of the library can use multiple contexts to either :
- Use multiple OpenCL devices (multi-GPU or CPU & GPU)
- Run multiple operations at a time on the same GPU (to get 100% usage)

void ocip_API ocipSetCLFilesPath(const char * Path);
Sets the path where the .cl files are located.
Calling this function is mandatory and must be done prior to preparing a program or to call a processing primitive.
It can be called before calling ocipInitialize

ocip_API const char * ocipGetErrorName(ocipError Error);
Returns the name of the error code

ocipError ocip_API ocipGetDeviceName(char * Name, uint BufferLength);
Returns the name of the device used by the given context

ocipError ocip_API ocipFinish();
Waits until all queued operations of the current context are done.
When this function returns, the device will have finished all operations previously issued on this context.


ocipError ocip_API ocipCreateImage(ocipImage * ImagePtr, SImage image, cl_mem_flags flags);
Creates an image on the device according to the information provided in the SImage

ocipError ocip_API ocipSendImage(ocipImage Image);
Sends the image to device memory

ocipError ocip_API ocipReadImage(ocipImage Image);
Reads the image from device memory

ocipError ocip_API ocipReleaseImage(ocipImage Image);
Releases the image


ocipError ocip_API ocipPrepare*(ocipImage Image);
Prepare for executing processing operations.
These functions prepare the processing for the given image (builds the OpenCL program). 
If ocipPrepare*() is not called before calling a processing primitive,
the first call to the primitive for a given image type will take a long time (likely >100ms).

ocipError ocip_API ocipPrepare*(ocipProgram * ProgramPtr, ocipImage Image);
These functions prepare the processing for the given image (builds the OpenCL program) and
also allocate any temporary buffers. Any calls to processing primitives that need a ocipProgram
argument must be done with a program handle created with the proper ocipPrepare*() and with the same
image as the one used when calling ocipPrepare*().

ocipError ocip_API ocipReleaseProgram(ocipProgram Program);
Releases a program previously created by a call to ocipPrepare*

Example :
ocipError ocip_API ocipPrepareImageArithmetic(ocipImage Image);
Prepares processing for arithmetic on the given type of images.


Processing functions 

The rest of the functions are processing functions
They will do the following :
Build the OpenCL program if it has not been built beforehand by calling ocipPrepare*()
Send the Source* images to the device if they have not been sent beforehand
   (if the image has been sent in the past but its content on the host has changed since then and that new data is needed in the device, please call ocipSendImage())
Add the processing operation to the queue
Most primitives then return before any processing has begun.

