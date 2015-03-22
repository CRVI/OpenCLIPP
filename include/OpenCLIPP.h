////////////////////////////////////////////////////////////////////////////////
//! @file	: OpenCLIPP.h 
//! @date   : Jul 2013
//!
//! @brief  : C Interface for the OpenCLIPP library
//! 
//! Copyright (C) 2013 - CRVI
//!
//! This file is part of OpenCLIPP.
//! 
//! OpenCLIPP is free software: you can redistribute it and/or modify
//! it under the terms of the GNU Lesser General Public License version 3
//! as published by the Free Software Foundation.
//! 
//! OpenCLIPP is distributed in the hope that it will be useful,
//! but WITHOUT ANY WARRANTY; without even the implied warranty of
//! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//! GNU Lesser General Public License for more details.
//! 
//! You should have received a copy of the GNU Lesser General Public License
//! along with OpenCLIPP.  If not, see <http://www.gnu.org/licenses/>.
//! 
////////////////////////////////////////////////////////////////////////////////

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include <CL/opencl.h>

// Definition of ocip_API - used by Microsoft compiler to generate a DLL
#ifdef _MSC_VER
#ifdef OPENCLIPP_EXPORTS
#define ocip_API __declspec(dllexport)
#else
#define ocip_API __declspec(dllimport)
#endif
#else
#define ocip_API
#endif // _MSC_VER

#ifdef __cplusplus
extern "C" {
#endif

typedef char ocipBool;   ///< Used to signify which function parameters are boolean
typedef unsigned int uint;    ///< Shortcut for the often used 'unsigend int'


#include <SImage.h>

/// The SImage structure is used to tell the library the type and size of images when creating image objects.
typedef struct SImage SImage;


/// Type used as return values of most ocip calls.
/// Successful calls will return CL_SUCCESS (0) while
/// unsuccesful calls will return a negative value.
/// Use ocipGetErrorName() to get the name of the error
typedef cl_int ocipError;

typedef struct _cl_context * ocipContext; ///< A handle to a context
typedef struct _cl_image   * ocipImage;   ///< A handle to an image in the device
typedef struct _cl_program * ocipProgram; ///< A handle to a program


/// Lists the possible interpolation types useable in some primitives
enum ocipInterpolationType
{
   ocipNearestNeighbour,   ///< Chooses the value of the closest pixel - Fastest
   ocipLinear,             ///< Does a bilinear interpolation of the 4 closest pixels
   ocipCubic,              ///< Does a bicubic interpolation of the 16 closest pixels
   ocipLanczos2,           ///< Does 2-lobed Lanczos interpolation using 16 pixels
   ocipLanczos3,           ///< Does 3-lobed Lanczos interpolation using 36 pixels
   ocipSuperSampling,      ///< Samples each pixel of the source for best resize result - for downsizing images only
   ocipBestQuality,        ///< Automatically selects the choice that will give the best image quality for the operation
};



/// Initialization.
/// Initializes OpenCL, creates an execution context, sets the new context as the current context
/// and returns the context handle.
/// The handle must be closed by calling ocipUninitialize when the context (or the whole library) is no longer needed.
/// ocipInitialize() can be called more than once, in that case, each context must be
/// released individually by a call to ocipUninitialize(). Images, Buffers and Programs
/// created from different context can't be mixed (a program can only run
/// with images or buffers that have been created from the same context).
/// \param ContextPtr : The value pointed by ContextPtr will be set to the new context handle
/// \param PreferredPlatform : Can be set to a specific platform (Ex: "Intel") and
///         that platform will be used if available. If the preferred platform is not
///         found or is not specified, the default OpenCL platform will be used.
///         Set to NULL to let OpenCL choose the best computing device available.
/// \param deviceType : can be used to specicy usage of a device type (Ex: CL_DEVICE_TYPE_GPU)
///         See cl_device_type for allowed values
///         Set to CL_DEVICE_TYPE_ALL to let OpenCL choose the best computing device available.
ocipError ocip_API ocipInitialize(ocipContext * ContextPtr, const char * PreferredPlatform, cl_device_type deviceType);

/// Uninitialization.
/// Releases the context.
/// \param Context : Handle to the context to uninitialize
ocipError ocip_API ocipUninitialize(ocipContext Context);


/// Change the current context.
/// Advanced users of the library can use multiple contexts to either :
/// - Use multiple OpenCL devices (multi-GPU or CPU & GPU)
/// - Run multiple operations at a time on the same GPU (to get 100% usage)
/// \param Context : The context to use for the next library calls
ocipError ocip_API ocipChangeContext(ocipContext Context);


/// Set the Path of .cl files.
/// It is necessary to call this functione before creating any program
/// \param Path : Full path where the .cl files are located
void ocip_API ocipSetCLFilesPath(const char * Path);


/// Returns the name of the error code
/// \param Error : An OpenCL error code
/// \return the name of the given error code
ocip_API const char * ocipGetErrorName(ocipError Error);


/// Returns the name of the device used by the given context
/// \param Name : Pointer to a buffer to receive a null terminated string that will contain the device name
/// \param BufferLength : Number of elements in the Name buffer
ocipError ocip_API ocipGetDeviceName(char * Name, uint BufferLength);


/// Waits until all queued operations of this context are done.
/// When this function returns, the device will have finished all operations previously issued on this context.
ocipError ocip_API ocipFinish();


// Images

/// Image creation.
/// Allocates memory on the device to store an image like the one in 'Image'
/// If ImageData is not NULL, the pointer value will be saved in the object and
/// then used in these situations :
///   - As source of data when calling ocipSendImage(),
///      meaning memory at that address will be read and then copied in the device memory
///   - As source of data when calling a processing function with this image as source, if
///      this image has never been sent to the device before.
///   - As destination of data when calling ocipReadImage(),
///      meaning memory at that address will be overwritten by the data from the device
///
/// If ImageData is NULL, the image will not be able to be Sent nor Read.
/// Creating an image this way is useful for itermediary results of multi-step calculations or
/// as temporary image for the processing functions that need them.
/// \param ImagePtr : The value pointed to by ImagePtr will be set to the handle of the new image
/// \param Image : A SImage structure describing the image
/// \param ImageData : A pointer to where the image data is located in the main memmory.
///      Can either be NULL (for a device-only image) or point to an image that fits with the description in Image.
/// \param flags : The type of device memory to use, allowed values : CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY
ocipError ocip_API ocipCreateImage( ocipImage * ImagePtr, SImage Image, void * ImageData, cl_mem_flags flags);


/// Sends the image to the device.
/// The image data will referenced by the pointer in the SImage structure given during image creation
/// will be transferred to the device memory.
/// This function executes asyncronously, meaning it will return quickly before the transfer is done.
/// The data pointer given during image creation must remain valid until the transfer is complete.
/// A send operation will be issued automatically when calling a processing function with the image as Source,
/// if the image has not been sent already.
/// Use ocipSendImage to send new image data from the host to the device when the image on the host has changed.
ocipError ocip_API ocipSendImage(   ocipImage Image);

/// Reads the image from the device.
/// The image in the device will be read into the memory pointed to by the pointer in the SImage structure
/// given during image creation.
/// This function will wait until all previous operations, including the read to be complete before returning.
/// So after this function returns, the image on the host will contain the result of the processing operations.
ocipError ocip_API ocipReadImage(   ocipImage Image);

/// Releases an image.
/// Releases the device memory for this image.
/// The ocipImage handle will no longer be valid.
ocipError ocip_API ocipReleaseImage(ocipImage Image);


/// Prepare for executing processing operations.
/// ocipPrepareExample() does nothing, it is a place holder for documentation about
/// ocipPrepare* functions that have a single argument.\n
/// These functions will do the following :
///  - Load the .cl file containing the desired program
///  - Initialize an object to hold that program
///  - Build the program for the specified image
///
/// Before calling one of these functions, ocipSetCLFilesPath must have been called with the proper path.\n
/// This description is good for all functions of these forms :\n
/// ocipPrepare*(ocipImage Image);\n
/// If ocipPrepare*() is not called before calling the primitive, the operations
/// liste above will be done during the first call to the primitive with that type of image,
/// meaning the first call could take many hundreds of miliseconds.
/// ocipPrepare*() can be called more than once to be ready for images of different types.
/// \param Image : The program will be built (prepared) for that image so that later calls to a processing operation of
///   that category will be fast.
ocipError ocip_API ocipPrepareExample(ocipImage Image);

/// Prepare for executing processing operations.
/// ocipPrepareExample2() does nothing, it is a place holder for documentation about
/// ocipPrepare* functions that take a ocipProgram * argument.\n
/// These functions will do the following :
///  - Load the .cl file containing the desired program
///  - Initialize an object to hold that program
///  - Build the program for the specified image
///  - Allocate and initialize temprary memory buffers needed for processing the given image
///
/// Before calling one of these functions, ocipSetCLFilesPath must have been called with the proper path.\n
/// This description is good for all functions of this form :\n
/// ocipPrepare*(ocipProgram * ProgramPtr, ocipImage Image[, ...]);\n
/// This version of ocipPrepare*() is for primtives that have a ocipProgram argument.
/// These primitives can't be called without a valid program.
/// This is for operations that need temporary buffers or other resources
/// to be allocated in advance. These preprare function will, in addition to building the program,
/// allocate these resources. More than one program handle can be prepared
/// to work with different images.
/// In these cases, ocipReleaseProgram must be called to close the program
/// handle when the program is no longer needed.
/// \param ProgramPtr : A pointer to a variable that will receive the program handle.
/// \param Image : The program will be built (prepared) for the given image.
///                Temporary buffers will also be pre-allocated with the proper size for that image.
ocipError ocip_API ocipPrepareExample2(ocipProgram * ProgramPtr, ocipImage Image);

/// Releases a program.
/// Releases the program, the given program handle will no longer be valid.
ocipError ocip_API ocipReleaseProgram(ocipProgram Program);


// Note about processing functions
// Most processing functions are asyncrhonous (non-blocking), meaning 
// they will return quickly before any device computation has been done.
// Many processing functions and Send operations can be issued and then the host
// is free to do other tasks in parralel to the computation.
// Use ocipRead* to wait for completion of all processing operations and transfer
// the result to the host or use ocipFinish() to wait for all queued operations.
// Some program's processing functions are synchronous. If they are, a comment before ocipPrepare*
// will describe their behaviour.
// The processing functions that take a ocipProgram argument  need to be used
// with the proper program, which is the one referenced by
// the handle generated by the ocipPrepare*() function declared above it.
// The first call to a primtive (or the first call with a different type of image)
// will take a long time because the program needs to be prepared for the image.
// To prevent this delay, call ocipPrepare*() with the image beforehand.



// Conversions -----------------------------------------------------------------------------------------
ocipError ocip_API ocipPrepareConversion(ocipImage Image);  ///< See ocipPrepareExample

/// From any image type to any image type - no value scaling
ocipError ocip_API ocipConvert(   ocipImage Source, ocipImage Dest);

/// From any image type to any image type - automatic value scaling.
/// Scales the input values by the ration of : output range/input range
/// The range is 0,255 for 8u, -128,127 for 8s, ...
/// The range is 0,1 for float
ocipError ocip_API ocipScale(     ocipImage Source, ocipImage Dest);

/// From any image type to any image type with given scaling.
/// Does the conversion Dest = (Source * Ratio) + Offset
ocipError ocip_API ocipScale2(    ocipImage Source, ocipImage Dest, int Offset, float Ratio);

/// Copies an image.
/// Both images must be of the same type.
ocipError ocip_API ocipCopy(      ocipImage Source, ocipImage Dest);


/// Converts a color (4 channel) image to a 1 channel image by averaging the first 3 channels
ocipError ocip_API ocipToGray(    ocipImage Source, ocipImage Dest);

/// Selects 1 channel from a 4 channel image to a 1 channel image - ChannelNo can be from 1 to 4
ocipError ocip_API ocipSelectChannel(ocipImage Source, ocipImage Dest, int ChannelNo);

/// Converts a 1 channel image to a 4 channel image - first 3 channels of Dest will be set to the value of the first channel of Source
ocipError ocip_API ocipToColor(   ocipImage Source, ocipImage Dest);


// Arithmetic --------------------------------------------------------------------------------------
ocipError ocip_API ocipPrepareArithmetic(ocipImage Image);     ///< See ocipPrepareExample
// Between two images
ocipError ocip_API ocipAdd(      ocipImage Source1, ocipImage Source2, ocipImage Dest);   ///< D = S1 + S2
ocipError ocip_API ocipAddSquare(ocipImage Source1, ocipImage Source2, ocipImage Dest);   ///< D = S1 + S2 * S2
ocipError ocip_API ocipSub(      ocipImage Source1, ocipImage Source2, ocipImage Dest);   ///< D = S1 - S2
ocipError ocip_API ocipAbsDiff(  ocipImage Source1, ocipImage Source2, ocipImage Dest);   ///< D = abs(S1 - S2)
ocipError ocip_API ocipMul(      ocipImage Source1, ocipImage Source2, ocipImage Dest);   ///< D = S1 * S2
ocipError ocip_API ocipDiv(      ocipImage Source1, ocipImage Source2, ocipImage Dest);   ///< D = S1 / S2
ocipError ocip_API ocipImgMin(   ocipImage Source1, ocipImage Source2, ocipImage Dest);   ///< D = min(S1, S2)
ocipError ocip_API ocipImgMax(   ocipImage Source1, ocipImage Source2, ocipImage Dest);   ///< D = max(S1, S1)
ocipError ocip_API ocipImgMean(  ocipImage Source1, ocipImage Source2, ocipImage Dest);   ///< D = (S1 + S2) / 2
ocipError ocip_API ocipCombine(  ocipImage Source1, ocipImage Source2, ocipImage Dest);   ///< D = sqrt(S1 * S1 + S2 * S2)

// Image and value
ocipError ocip_API ocipAddC(    ocipImage Source, ocipImage Dest, float value);   ///< D = S + v
ocipError ocip_API ocipSubC(    ocipImage Source, ocipImage Dest, float value);   ///< D = S - v
ocipError ocip_API ocipAbsDiffC(ocipImage Source, ocipImage Dest, float value);   ///< D = abs(S - v)
ocipError ocip_API ocipMulC(    ocipImage Source, ocipImage Dest, float value);   ///< D = S * v
ocipError ocip_API ocipDivC(    ocipImage Source, ocipImage Dest, float value);   ///< D = S / v
ocipError ocip_API ocipRevDivC( ocipImage Source, ocipImage Dest, float value);   ///< D = v / S
ocipError ocip_API ocipMinC(    ocipImage Source, ocipImage Dest, float value);   ///< D = min(S, v)
ocipError ocip_API ocipMaxC(    ocipImage Source, ocipImage Dest, float value);   ///< D = max(S, v)
ocipError ocip_API ocipMeanC(   ocipImage Source, ocipImage Dest, float value);   ///< D = (S + V) / 2

// Calculation on one image
ocipError ocip_API ocipAbs(    ocipImage Source, ocipImage Dest);   ///< D = abs(S)
ocipError ocip_API ocipInvert( ocipImage Source, ocipImage Dest);   ///< D = 255 - S
ocipError ocip_API ocipSqr(    ocipImage Source, ocipImage Dest);   ///< D = S * S

// Calculation on one image - float required
ocipError ocip_API ocipExp(    ocipImage Source, ocipImage Dest);   ///< D = exp(S)
ocipError ocip_API ocipLog(    ocipImage Source, ocipImage Dest);   ///< D = log(S)
ocipError ocip_API ocipSqrt(   ocipImage Source, ocipImage Dest);   ///< D = sqrt(S)
ocipError ocip_API ocipSin(    ocipImage Source, ocipImage Dest);   ///< D = sin(S)
ocipError ocip_API ocipCos(    ocipImage Source, ocipImage Dest);   ///< D = cos(S)



// Logic -------------------------------------------------------------------------------------------
ocipError ocip_API ocipPrepareLogic(ocipImage Image); ///< See ocipPrepareExample
// Bitwise operations - float images not allowed
ocipError ocip_API ocipAnd( ocipImage Source1, ocipImage Source2, ocipImage Dest);   ///< D = S1 & S2
ocipError ocip_API ocipOr(  ocipImage Source1, ocipImage Source2, ocipImage Dest);   ///< D = S1 | S2
ocipError ocip_API ocipXor( ocipImage Source1, ocipImage Source2, ocipImage Dest);   ///< D = S1 ^ S2
ocipError ocip_API ocipAndC(ocipImage Source, ocipImage Dest, uint value);            ///< D = S & v
ocipError ocip_API ocipOrC( ocipImage Source, ocipImage Dest, uint value);            ///< D = S | v
ocipError ocip_API ocipXorC(ocipImage Source, ocipImage Dest, uint value);            ///< D = S ^ v
ocipError ocip_API ocipNot( ocipImage Source, ocipImage Dest);                        ///< D = ~S



// LUT ---------------------------------------------------------------------------------------------
ocipError ocip_API ocipPrepareLUT(ocipImage Image);   ///< See ocipPrepareExample

/// Performs a LUT operation.
/// levels and values must be arrays of NbValues elements
/// Dest will contain the following transformation :
/// find value v where (S >= levels[v] && S < levels[v + 1])
/// D = values[v]
/// \param levels : Array of size NbValues describing the levels to look at in Source
/// \param values : Array of size NbValues describing the values to use for those levels
ocipError ocip_API ocipLut(       ocipImage Source, ocipImage Dest, uint * levels, uint * values, int NbValues);

/// Performs a linear LUT operation.
/// levels and values must be arrays of NbValues elements
/// Dest will contain the following transformation :
/// find value v where (S >= levels[v] && S < levels[v + 1])
/// ratio = (S - levels[v]) / (levels[v + 1] - levels[v])
/// D = values[v] + (values[v + 1] - values[v]) * ratio
/// \param levels : Array of size NbValues describing the levels to look at in Source
/// \param values : Array of size NbValues describing the values to use for those levels
ocipError ocip_API ocipLutLinear( ocipImage Source, ocipImage Dest, float * levels, float * values, int NbValues);

/// Performs a LUT on 8 bit unsigned images.
/// D = values[S]
ocipError ocip_API ocipBasicLut(  ocipImage Source, ocipImage Dest, unsigned char * values);

/// Scales values of Source image according to the given input and output ranges
ocipError ocip_API ocipLutScale(     ocipImage Source, ocipImage Dest, float SrcMin, float SrcMax, float DstMin, float DstMax);



// Morphology --------------------------------------------------------------------------------------
ocipError ocip_API ocipPrepareMorphology(ocipImage Image);  ///< See ocipPrepareExample
// Single iteration
ocipError ocip_API ocipErode(    ocipImage Source, ocipImage Dest, int Width);      ///< 1 Iteration
ocipError ocip_API ocipDilate(   ocipImage Source, ocipImage Dest, int Width);      ///< 1 Iteration
ocipError ocip_API ocipGradient( ocipImage Source, ocipImage Dest, ocipImage Temp, int Width);  ///< Dilate - Erode
// Multiple iterations
ocipError ocip_API ocipErode2(   ocipImage Source, ocipImage Dest, ocipImage Temp, int Iterations, int Width);
ocipError ocip_API ocipDilate2(  ocipImage Source, ocipImage Dest, ocipImage Temp, int Iterations, int Width);
ocipError ocip_API ocipOpen(     ocipImage Source, ocipImage Dest, ocipImage Temp, int Depth, int Width);   ///< Erode then dilate
ocipError ocip_API ocipClose(    ocipImage Source, ocipImage Dest, ocipImage Temp, int Depth, int Width);   ///< Dilate then erode
ocipError ocip_API ocipTopHat(   ocipImage Source, ocipImage Dest, ocipImage Temp, int Depth, int Width);   ///< Source - Open
ocipError ocip_API ocipBlackHat( ocipImage Source, ocipImage Dest, ocipImage Temp, int Depth, int Width);   ///< Close - Source



// Transformations ---------------------------------------------------------------------------------
ocipError ocip_API ocipPrepareTransform(ocipImage Image);   ///< See ocipPrepareExample

/// Mirrors the image along X.
/// D(x,y) = D(width - x - 1, y)
ocipError ocip_API ocipMirrorX(   ocipImage Source, ocipImage Dest);

/// Mirrors the image along Y.
/// D(x,y) = D(x, height - y - 1)
ocipError ocip_API ocipMirrorY(   ocipImage Source, ocipImage Dest);

/// Flip : Mirrors the image along X and Y.
/// D(x,y) = D(width - x - 1, height - y - 1)
ocipError ocip_API ocipFlip(      ocipImage Source, ocipImage Dest);

/// Transposes the image.
/// Dest must have a width >= as Source's height and a height >= as Source's width
/// D(x,y) = D(y, x)
ocipError ocip_API ocipTranspose( ocipImage Source, ocipImage Dest);

/// Rotates the source image aroud the origin (0,0) and then shifts it.
/// \param Source : Source image
/// \param Dest : Destination image
/// \param Angle : Angle to use for the rotation, in degrees.
/// \param XShift : Shift along horizonltal axis to do after the rotation.
/// \param YShift : Shift along vertical axis to do after the rotation.
/// \param Interpolation : Type of interpolation to use.
///      Available choices are : NearestNeighbour, Linear, Cubic or BestQuality
///      BestQuality will use Cubic.
ocipError ocip_API ocipRotate(    ocipImage Source, ocipImage Dest, double Angle, double XShift, double YShift, enum ocipInterpolationType Interpolation);

/// Resizes the image.
/// \param Source : Source image
/// \param Dest : Destination image
/// \param Interpolation : Type of interpolation to use.
///      Available choices are : NearestNeighbour, Linear, Cubic or BestQuality
///      BestQuality will use linear when shrinking and Cubic when enlarging.
/// \param KeepRatio : If false, Dest will be filled with the image from source, potentially changing
///      the aspect ratio of the image. \n If true, the aspect ratio of the image will be kept, potentially
///      leaving part of Dest with invalid (unchaged) data to the right or to the bottom.
ocipError ocip_API ocipResize(    ocipImage Source, ocipImage Dest, enum ocipInterpolationType Interpolation, ocipBool KeepRatio);

/// Shearing transformation.
/// \param Source : Source image
/// \param Dest : Destination image
/// \param ShearX : X Shearing coefficient.
/// \param ShearY : Y Shearing coefficient.
/// \param XShift : Shift along horizonltal axis to do after the shearing.
/// \param YShift : Shift along vertical axis to do after the shearing.
/// \param Interpolation : Type of interpolation to use.
///      Available choices are : NearestNeighbour, Linear, Cubic or BestQuality
///      BestQuality will use Cubic.
ocipError ocip_API ocipShear(ocipImage Source, ocipImage Dest, double ShearX, double ShearY, double XShift, double YShift, enum ocipInterpolationType Interpolation);

/// Remap
/// \param Source : Source image
/// \param MapX : X Map image, must be 1 channel, F32
/// \param MapY : Y Map image, must be 1 channel, F32
/// \param Dest : Destination image
/// \param Interpolation : Type of interpolation to use.
///      Available choices are : NearestNeighbour, Linear, Cubic or BestQuality
///      BestQuality will use Cubic.
ocipError ocip_API ocipRemap(ocipImage Source, ocipImage MapX, ocipImage MapY, ocipImage Dest, enum ocipInterpolationType Interpolation);

/// Sets all values of Dest to value
ocipError ocip_API ocipSet(       ocipImage Dest, float Value);



// Filters -----------------------------------------------------------------------------------------
ocipError ocip_API ocipPrepareFilters(ocipImage Image);  ///< See ocipPrepareExample

/// Gaussian blur filter - with sigma parameter.
/// \param Sigma : Intensity of the filer - Allowed values : 0.01-10
ocipError ocip_API ocipGaussianBlur(ocipImage Source, ocipImage Dest, float Sigma);

/// Gaussian filter - with width parameter.
/// \param Width : Width of the filter box - Allowed values : 3 or 5
ocipError ocip_API ocipGauss(     ocipImage Source, ocipImage Dest, int Width);

/// Sharpen filter.
/// \param Width : Width of the filter box - Allowed values : 3
ocipError ocip_API ocipSharpen(   ocipImage Source, ocipImage Dest, int Width);

/// Smooth filter - or Box filter.
/// \param Width : Width of the filter box - Allowed values : Impair & >=3
ocipError ocip_API ocipSmooth(    ocipImage Source, ocipImage Dest, int Width);

/// Median filter
/// \param Width : Width of the filter box - Allowed values : 3 or 5
ocipError ocip_API ocipMedian(    ocipImage Source, ocipImage Dest, int Width);

/// Vertical Sobel filter
/// \param Width : Width of the filter box - Allowed values : 3 or 5
ocipError ocip_API ocipSobelVert( ocipImage Source, ocipImage Dest, int Width);

/// Horizontal Sobel filter
/// \param Width : Width of the filter box - Allowed values : 3 or 5
ocipError ocip_API ocipSobelHoriz(ocipImage Source, ocipImage Dest, int Width);

/// Cross Sobel filter
/// \param Width : Width of the filter box - Allowed values : 3 or 5
ocipError ocip_API ocipSobelCross(ocipImage Source, ocipImage Dest, int Width);

/// Combined Sobel filter
/// Does SobelVert & SobelHoriz and the combines the two with sqrt(V*V + H*H)
/// \param Width : Width of the filter box - Allowed values : 3 or 5
ocipError ocip_API ocipSobel(     ocipImage Source, ocipImage Dest, int Width);

/// Vertical Prewitt filter
/// \param Width : Width of the filter box - Allowed values : 3 or 5
ocipError ocip_API ocipPrewittVert(ocipImage Source, ocipImage Dest, int Width);

/// Horizontal Prewitt filter
/// \param Width : Width of the filter box - Allowed values : 3 or 5
ocipError ocip_API ocipPrewittHoriz(ocipImage Source, ocipImage Dest, int Width);

/// Combined Prewitt filter
/// Does PrewittVert & PrewittHoriz and the combines the two with sqrt(V*V + H*H)
/// \param Width : Width of the filter box - Allowed values : 3 or 5
ocipError ocip_API ocipPrewitt(     ocipImage Source, ocipImage Dest, int Width);

/// Vertical Scharr filter
/// \param Width : Width of the filter box - Allowed values : 3 or 5
ocipError ocip_API ocipScharrVert(   ocipImage Source, ocipImage Dest, int Width);

/// Horizontal Scharr filter
/// \param Width : Width of the filter box - Allowed values : 3 or 5
ocipError ocip_API ocipScharrHoriz(  ocipImage Source, ocipImage Dest, int Width);

/// Combined Scharr filter
/// Does ScharrVert & ScharrHoriz and the combines the two with sqrt(V*V + H*H)
/// \param Width : Width of the filter box - Allowed values : 3 or 5
ocipError ocip_API ocipScharr(       ocipImage Source, ocipImage Dest, int Width);

/// Hipass filter
/// \param Width : Width of the filter box - Allowed values : 3 or 5
ocipError ocip_API ocipHipass(       ocipImage Source, ocipImage Dest, int Width);

/// Laplace filter
/// \param Width : Width of the filter box - Allowed values : 3 or 5
ocipError ocip_API ocipLaplace(      ocipImage Source, ocipImage Dest, int Width);



// Histogram ---------------------------------------------------------------------------------------
// All Histogram operations are Syncrhonous, meaning they block until the histogram is calculated and set to Histogram
ocipError ocip_API ocipPrepareHistogram(ocipImage Image);   ///< See ocipPrepareExample

/// Calculates the Histogram of the first channel of the image
/// \param Histogram : Array of 256 elements that will receive the histogram values
ocipError ocip_API ocipHistogram_1C(ocipImage Source, uint * Histogram);

/// Calculates the Histogram of all channels of the image
/// \param Histogram : Array of 1024 elements that will receive the histogram values
ocipError ocip_API ocipHistogram_4C(ocipImage Source, uint * Histogram);

/// Calculates the Otsu threshold given an histogram
ocipError ocip_API ocipOtsuThreshold(ocipImage Source, uint * Value);



// Statistics --------------------------------------------------------------------------------------
// All Statistics operations are Syncrhonous, meaning they block until the value is calculated and set to Result
ocipError ocip_API ocipPrepareStatistics(ocipProgram * ProgramPtr, ocipImage Image);   ///< See ocipPrepareExample2
// Result must point to an array that is at least NbChannels long
ocipError ocip_API ocipMin(             ocipProgram Program, ocipImage Source, double * Result);                  ///< Finds the minimum value in the image
ocipError ocip_API ocipMax(             ocipProgram Program, ocipImage Source, double * Result);                  ///< Finds the maximum value in the image
ocipError ocip_API ocipMinAbs(          ocipProgram Program, ocipImage Source, double * Result);                  ///< Finds the minimum of the absolute of the values in the image
ocipError ocip_API ocipMaxAbs(          ocipProgram Program, ocipImage Source, double * Result);                  ///< Finds the maxumum of the absolute of the values in the image
ocipError ocip_API ocipSum(             ocipProgram Program, ocipImage Source, double * Result);                  ///< Calculates the sum of all pixel values
ocipError ocip_API ocipSumSqr(          ocipProgram Program, ocipImage Source, double * Result);                  ///< Calculates the sum of the sqaure of all pixel values
ocipError ocip_API ocipMean(            ocipProgram Program, ocipImage Source, double * Result);                  ///< Calculates the mean value of all pixel values
ocipError ocip_API ocipMeanSqr(         ocipProgram Program, ocipImage Source, double * Result);                  ///< Calculates the mean of the square of all pixel values
ocipError ocip_API ocipStdDev(          ocipProgram Program, ocipImage Source, double * Result);                  ///< Calculates the standard deviation of all pixel values
ocipError ocip_API ocipMean_StdDev(     ocipProgram Program, ocipImage Source, double * Mean, double * StdDev);   ///< Calculates the standard deviation and mean of all pixel values
// These operate only on the first channel of the image, Result, IndexX and IndexY can point to a single value
ocipError ocip_API ocipCountNonZero(    ocipProgram Program, ocipImage Source, uint   * Result);                              ///< Calculates the number of pixels that have a non zero value
ocipError ocip_API ocipMinIndx(         ocipProgram Program, ocipImage Source, double * Result, int * IndexX, int * IndexY);  ///< Finds the minimum value in the image and the coordinate (index) of the pixel with that value
ocipError ocip_API ocipMaxIndx(         ocipProgram Program, ocipImage Source, double * Result, int * IndexX, int * IndexY);  ///< Finds the maximum value in the image and the coordinate (index) of the pixel with that value
ocipError ocip_API ocipMinAbsIndx(      ocipProgram Program, ocipImage Source, double * Result, int * IndexX, int * IndexY);  ///< Finds the minimum of the absolute values in the image and the coordinate (index) of the pixel with that value
ocipError ocip_API ocipMaxAbsIndx(      ocipProgram Program, ocipImage Source, double * Result, int * IndexX, int * IndexY);  ///< Finds the maximum of the absolute values in the image and the coordinate (index) of the pixel with that value



// Thresholding ------------------------------------------------------------------------------------
ocipError ocip_API ocipPrepareThresholding(ocipImage Image); ///< See ocipPrepareExample

enum ECompareOperation { LT, LQ, EQ, GQ, GT, };

/// D = (S Op Thresh ? value : S)
ocipError ocip_API ocipThreshold(    ocipImage Source,  ocipImage Dest, float Thresh, float value, enum ECompareOperation Op);

/// D = (S > threshGT ? valueHigher : (S < threshLT ? valueLower : S) )
ocipError ocip_API ocipThresholdGTLT(ocipImage Source,  ocipImage Dest, float threshLT, float valueLower, float threshGT, float valueHigher);

/// D = (S1 Op S2 ? S1 : S2)
ocipError ocip_API ocipThreshold_Img(ocipImage Source1, ocipImage Source2, ocipImage Dest, enum ECompareOperation Op);

/// D = (S Op S2) - D will be 0 or 255
/// Dest must be U8 and 1 channel
ocipError ocip_API ocipCompare(      ocipImage Source1, ocipImage Source2, ocipImage Dest, enum ECompareOperation Op);

/// D = (S1 Op V) - D will be 0 or 255
/// Dest must be U8 and 1 channel
ocipError ocip_API ocipCompareC(      ocipImage Source,  ocipImage Dest, float Value, enum ECompareOperation Op);



// Blobs -------------------------------------------------------------------------------------------
// All Blob operations are Syncrhonous, meaning they block until the computation is complete but
// no read operation is issued. ocipReadImage() must be called to transfer the result into host memory.
ocipError ocip_API ocipPrepareBlob(ocipProgram * ProgramPtr, ocipImage Image);    ///< See ocipPrepareExample2

/// Compute the blob labels for the given image.
/// PrepareFor() must be called with the same Source image before calling ComputeLabels()
/// All non-zero pixels will be grouped with their neighbours and given a label number
/// After calling, Labels image will contain the label values for each pixel,
/// and -1 (or 0xffffffff) for pixels that were 0
/// \param Source : The image to analyze
/// \param Labels : must be a 32b integer image
/// \param ConnectType : Type of pixel connectivity, can be 4 or 8
ocipError ocip_API ocipComputeLabels(ocipProgram Program, ocipImage Source, ocipImage Labels, int ConnectType);

/// Renames the labels to be from 0 to NbLabels-1.
/// \param Labels : must be an image resulting from a previous call to ComputeLabels()
ocipError ocip_API ocipRenameLabels(ocipProgram Program, ocipImage Labels);



// FFT ---------------------------------------------------------------------------------------------
ocipError ocip_API ocipPrepareFFT(  ocipProgram * ProgramPtr,   ocipImage RealImage,       ocipImage ComplexImage);   ///< See ocipPrepareExample2

ocipBool  ocip_API ocipIsFFTAvailable();  ///< Returns true if the library has been compiled with FFT

/// Forward Fast Fourrier Transform.
/// Executes a fast fourrier transform on the given image
/// The size of RealSource is used as the dimention for the transformation.
/// \param Program : The FFT program prepared by ocipPrepareFFT
/// \param RealSource : An image containing a 1 channel image of F32 real values
/// \param ComplexDest : An image that will received the transformed image as complex numbers.
///                      It must be 2 channels of F32 and its width must be >= Width(RealSource)/2+1\n
///                      First channel is Real and second channel is Imaginary
ocipError ocip_API ocipFFTForward(  ocipProgram Program,        ocipImage RealSource,      ocipImage ComplexDest);

/// Inverse (Backward) Fast Fourrier Transform.
/// Executes an inverse fast fourrier transform on the given complex image
/// The size of RealDest is used as the dimention for the transformation.
/// \param Program : The FFT program prepared by ocipPrepareFFT
/// \param ComplexSource : An image containing a 2 channel image of F32 as complex numbers. First channel is Real and second channel is Imaginary.
///                        Its width must be >= Width(RealDest)/2+1.
/// \param RealDest : An image containing a 1 channel image of F32. Will receive the transformed image as real numbers only (no imaginary part).
ocipError ocip_API ocipFFTInverse( ocipProgram Program,        ocipImage ComplexSource,   ocipImage RealDest);



// Integral ----------------------------------------------------------------------------------------
ocipError ocip_API ocipPrepareIntegral(ocipProgram * ProgramPtr, ocipImage Image); ///< See ocipPrepareExample

///< Scans the image and generates the Integral sum into Dest buffer - Dest must be F32 or F64 - 1 channel
ocipError ocip_API ocipIntegral( ocipProgram Program, ocipImage Source, ocipImage Dest);

///< Scans the image and generates the Square Integral sum into Dest buffer - Dest must be F32 or F64 - 1 channel
ocipError ocip_API ocipSqrIntegral( ocipProgram Program, ocipImage Source, ocipImage Dest);



// Image Proximity ---------------------------------------------------------------------------------
// All ImageProximity operations are Syncrhonous, meaning they block until the ImageProximity is calculated and set to the result
// Use only small template images (<=16x16 pixels)
// Will be very slow if big template images are used
// For faster image proximity operations with big template image, use ImageProximityFFT
ocipError ocip_API ocipPrepareProximity(ocipImage Image);   ///< See ocipPrepareExample

/// Computes normalized Euclidean distance between an image and a template.
ocipError ocip_API ocipSqrDistance_Norm(ocipImage Source, ocipImage Template, ocipImage Dest);

/// Computes Euclidean distance between an image and a template.
ocipError ocip_API ocipSqrDistance(ocipImage Source, ocipImage Template, ocipImage Dest);

//Computes the sum of the absolute difference between an image and a tamplate
ocipError ocip_API ocipAbsDistance(ocipImage Source, ocipImage Template, ocipImage Dest);

//Computes normalized cross-correlation between an image and a template.
ocipError ocip_API ocipCrossCorr(ocipImage Source, ocipImage Template, ocipImage Dest);

//Computes normalized the cross-correlation between an image and a tamplate
ocipError ocip_API ocipCrossCorr_Norm(ocipImage Source, ocipImage Template, ocipImage Dest);



// Image Proximity accelerated using FFT -----------------------------------------------------------
// If Template is small (<16x16 pixels), the standard versions above may be faster
// FFT operations do not work on images bigger than 16.7Mpixels
ocipError ocip_API ocipPrepareImageProximityFFT(ocipProgram * ProgramPtr, ocipImage Image, ocipImage Template);

/// Square different template matching - Images must be F32 - 1 channel
ocipError ocip_API ocipSqrDistanceFFT(ocipProgram Program, ocipImage Source, ocipImage Template, ocipImage Dest);

/// Normalized square different template matching - Images must be F32 - 1 channel
ocipError ocip_API ocipSqrDistanceFFT_Norm(ocipProgram Program, ocipImage Source, ocipImage Template, ocipImage Dest);

/// Cross correlation template matching - Images must be F32 - 1 channel
ocipError ocip_API ocipCrossCorrFFT(ocipProgram Program, ocipImage Source, ocipImage Template, ocipImage Dest);

/// Cross correlation template matching - Images must be F32 - 1 channel
ocipError ocip_API ocipCrossCorrFFT_Norm(ocipProgram Program, ocipImage Source, ocipImage Template, ocipImage Dest);


#ifdef __cplusplus
}
#endif
