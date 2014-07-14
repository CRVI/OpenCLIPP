
This library is designed to do accelerated image processing using OpenCL

It is designed to be simple to use and to have a low overhead.

How to build the library :

Windows : Use the .sln file with Visual Studio 2012

Other platforms : use make with the provided makefiles
   user@machine ~/OpenCLIPP# make

How to use the library :

#include <OpenCLIPP.hpp>

using namespace OpenCLIPP;

COpenCL CL;    // Initialize OpenCL

// Fill a SImage structure with image information
SImage SomeImage = {width, height, step, depth, channels};

// Create an image in the device
//  Image data is not sent to the device
Image Img1(CL, SomeImage, Img1Data);

// Create a second image in the device
Image Img2(CL, SomeImage, Img2Data);

// Create an OpenCL program to do arithmetic operations
Arithmetic arithmetic(CL);

// Will add 10 to each pixel value and put the result in Img2 in the device
//  Img1 is automatically sent to the device if it was not sent previously
//  The memory transfer and calculation are done asynchronously
arithmetic.Add(Img1, Img2, 10);

// Ask to read Img2 back into our image, after previous operations are done
//  true as argument means we want to wait for the transfer to be done
//  because the transfer will happen only after previous operations are done
Img2.Read(true);

// The image now contains the result of the addition


- Error handling

Any error or out of the ordinary event will be signaled by throwing a cl::Error object
The cl::Error contains the OpenCL status code ( .err() ) and a string representing the function that did the thow or an explanation of why it was thrown ( .what() ).

Using try & catch, either for the whole application or for each interaction with the library is recommended.

Try catch example :
try
{
   COpenCL CL;
   
   // More code
}
catch (cl::Error e)
{
   std::string Msg = "Eror : ";
   
   Msg += COpenCL::ErrorName(e.Err());
   Msg += " - ";
   Msg += e.what();
   
   // Use Msg
}


- List of Structures

SImage - List necessary information about an image
         It is used to initialize Image objects

   struct SImage
   {
      uint Width;
		uint Height;
		uint Step;           ///< Nb of bytes between each row
		uint Channels;       ///< Allowed values : 1, 2, 3 or 4

		enum EDataType
		{
			U8,
			S8,
			U16,
			S16,
			U32,
			S32,
			F32,
		} Type;
   };



- List of classes

COpenCL - Takes care of initializing OpenCL

   members :
      COpenCL(const char * PreferredPlatform = "", cl_device_type deviceType = CL_DEVICE_TYPE_ALL)
      
         Initializes OpenCL
         PreferredPlatform is the name of the OpenCL platform to use
         cl_device_type is the type of device to use, can be :
            CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ACCELERATOR,
            CL_DEVICE_TYPE_ALL or CL_DEVICE_TYPE_DEFAULT
         cl_device_type will be used only if PreferredPlatform is specified
         The default OpenCL platform will be used if no PreferredPlatform is given
         Currently, Only a single device is used and with an in-order command queue
         
      static void SetClFilesPath(const char * Path);
      
			Tells the library where the .cl files are located.
			It must be called with the full path of "OpenCLIPP/cl-files/".
			.cl file location must be specified before creating any program.
			The path must not contain spaces if using the NVIDIA platform.

		std::string GetDeviceName() const;
		
			Returns the name of the OpenCL device
			
		cl_device_type GetDeviceType() const;
		
			Returns the type of the OpenCL device
			
		EPlatformType GetPlatformType() const
		
			Returns the OpenCL platform type
			
      cl::CommandQueue& GetQueue()
      
         Retreives the command queue
         
      static const char * ErrorName(cl_int status)
      
         Returns a short textual description of the error code
         
      
      
Memory - base class for Buffers and Images - not useable directly

   members :
      bool IsInDevice() const
      
         Returns true if the memory object contains useful data
            Will return true if data has been sent to this object or if a kernel has written to this object
      
      void SetInDevice(bool inDevice = true)
      
         Sets or clears the InDevice state of the object

      virtual void SendIfNeeded()
      
         Overridable method that sends the assotiated host data to the device memory
         (does nothing by default as there is no assotiated data)


IBuffer : public Memory - base class for Buffers - not useable directly

   members :
      size_t Size() const
      
         returns the number of bytes in the buffer
         
         
TempBuffer : public IBuffer - Represents a buffer that is only in the device

   members :
      TempBuffer(COpenCL& CL, size_t size, cl_mem_flags flags = CL_MEM_READ_WRITE)
      
         Allocates memory on the device
            CL : The device to use
            size : Number of bytes to allocate
            flags : Type of device memory to allocate
               Allowed values :
                  CL_MEM_READ_WRITE - kernels can read & write values
                  CL_MEM_WRITE_ONLY - kernels can only write values
                  CL_MEM_READ_ONLY  - kernels can only read values
                  
ReadBuffer : public IBuffer - Represents a read only buffer

   members :
      template<class T>
      ReadBuffer(COpenCL& CL, T * data, size_t length)
      
         Creates a copy of the contents of data then allocates memory on the device
         Useful when using a small temporary buffer as parameter to a kernel that is launched asyncrhonously
            CL : The device to use
            length : Number of elements to allocate
            

Buffer : public IBuffer - Represents a buffer that exist in both the host and the device

   members :
      template<class T>
      Buffer(COpenCL& CL, T * data, size_t length, cl_mem_flags flags = CL_MEM_READ_WRITE);
      
         Allocates memory on the device, stores the address of the data
         Upon creation, the content of data is not copied and it is not transfered to the device
            CL : The device to use
            data : The host buffer - pointer must remain valid as long as Read() and Send() may be done on this object
            length : Number of elements to allocate
            flags : Type of device memory to allocate
               Allowed values :
                  CL_MEM_READ_WRITE - kernels can read & write values
                  CL_MEM_WRITE_ONLY - kernels can only write values
                  CL_MEM_READ_ONLY  - kernels can only read values

      void Read(bool blocking = false)
      
         Reads the data from buffer in the device to the host buffer given to the constructor
            blocking : if true, the method will wait for all device operations to be done, including the reading the data
         
      void Send(bool blocking = false)
      
         Sends the data from the host buffer to the buffer in the device
            blocking : if true, the method will wait for all device operations to be done

      virtual void SendIfNeeded()
      
         Calls Send() if IsInDevice() is false


ImageBase - base class for Images - not useable directly

   members:
      cl::NDRange FullRange()
         returns a global range that will start 1 work-item per pixel of the image
         
      cl::NDRange VectorRange(int NbElementsPerWorker)
         returns a global range that will start multiple work-items per pixel channel of the image
         
      size_t Width() const;
      size_t Height() const;
      size_t Step() const;
      size_t Depth() const;
      size_t NbChannels() const;
      size_t NbBytes() const;
      bool IsFloat() const;
      bool IsUnsigned() const;
         return information about the image
         
      SImage ToSImage() const;
         Creates a SImage that represents the image
         Only the size and type are used, the Data field of SImage will be null


Image : public Buffer, public ImageBase - Represents a buffer in the device that contains an image

   members:
      Image(COpenCL& CL, const SImage& image, cl_mem_flags flags = CL_MEM_READ_WRITE)
      
         Allocates memory in the device, with enough space to contain the image
         Upon creation, the content of image is not copied and it is not transfered to the device
            CL : The device to use
            image : Information about an image - pointer image.Data must remain valid as long as Read() and Send() may be done on this object
                    image can have 1, 3 or 4 channels
            flags : Type of device memory to allocate
               Allowed values :
                  CL_MEM_READ_WRITE - kernels can read & write values
                  CL_MEM_WRITE_ONLY - kernels can only write values
                  CL_MEM_READ_ONLY  - kernels can only read values

   important inherited members:
      void Read(bool blocking = false)
      
         Reads the data from the image in the device to the host image given to the constructor
            blocking : if true, the method will wait for all device operations to be done, including the reading the data
         
      void Send(bool blocking = false)
      
         Sends the data from the host image to the image in the device
            blocking : if true, the method will wait for all device operations to be done

      virtual void SendIfNeeded()
      
         Calls Send() if IsInDevice() is false
         
         
TempImage : public Image - Represents a device-only buffer that contains an image

   members:
   
      TempImage(COpenCL& CL, const SImage& image, cl_mem_flags flags = CL_MEM_READ_WRITE)
      
         Allocates memory in the device, with enough space to contain the image
            CL : The device to use
            image : Information about an image
            flags : Type of device memory to allocate
               Allowed values :
                  CL_MEM_READ_WRITE - kernels can read & write values
                  CL_MEM_WRITE_ONLY - kernels can only write values
                  CL_MEM_READ_ONLY  - kernels can only read values

      TempImage(COpenCL& CL, const SImage& image, size_t Depth, size_t Channels, bool isFloat = false)
      
         Allocates memory in the device, with enough space to contain an image that has the same dimentions as image but with a different data type
            CL : The device to use
            image : Information about an image, image can have 1, 3 or 4 channels
            Depth : number of bits per channel
            Channels : number of channels per pixel
            isFloat : true if we desire a float image

      virtual void SendIfNeeded()
      
         Does nothing (resides only in the device and can't be sent)


Program - Represents an OpenCL C program

   members:

      Program(COpenCL& CL, const char * Source, const char * options = "",
         const char * Path = "", bool build = false)
         
         Saves the given values for later use
            CL : The device to use
            Source : Source code for the program, can be ""
            options : A string listing compiler options
            Path : Complete path to the .cl file to use
         NOTE : At least one of these two parameters must not be empty : Source and Path

      bool Build()
      
         Builds the program if it has not yet been built
         If Source was "", the content of the file given with Path is loaded and used to build the program
         If Source contains a program, Path is ignored and the program in Source is built
         If the build fails, build information is written to std::cerr and a cl::Error is thrown
         Returns true if the build is successful or if the program has been built already
         
         
MultiProgram - Holder of multiple version of an OpenCL program with different compiler options - not useable directly
         
         
ImageProgram : public MultiProgram - Contains multiple versions of one program, a version for each supported data type :
   S8, U8, S16, U16, S32, U32, F32
   
   members:
   
      ImageProgram(COpenCL& CL, const char * Path)
      
         Prepares a program with the .cl file at the given path
         
      void PrepareFor(ImageBase& Source)
      
         Builds the proper version of the program for the given image
         
      Program& SelectProgram(ImageBase& Source)
      
         Builds and returns the proper version of the program for the given image

VectorProgram : public MultiProgram - Contains multiple versions of one program, three versions for each supported data type :
   S8, U8, S16, U16, S32, U32, F32
   
   members:
   
      VectorProgram(COpenCL& CL, const char * Path)
      
         Prepares a program with the .cl file at the given path
         
      void PrepareFor(ImageBase& Source)
      
         Builds the proper version of the program for the given image
         
      Program& SelectProgram(ImageBase& Source)
      
         Builds and returns the proper version of the program for the given image

