
template<typename DataType> class RemapBench;
template<typename DataType> class RemapLinearBench;
template<typename DataType> class RemapCubicBench;


template<typename DataType>
class RemapBench : public BenchUnaryBase<DataType>
{
public:
   RemapBench(ocipInterpolationType Interpolation = ocipNearestNeighbour)
   :  m_Interpolation(Interpolation),
      m_CLMapX(nullptr),
      m_CLMapY(nullptr),
      m_NPPMapX(nullptr),
      m_NPPMapY(nullptr)
   { }

   void RunIPP();
   void RunCL();
   void RunNPP();
   void RunCV();

   float CompareTolerance() const
   {
      if (is_same<DataType, float>::value)
         return 0.01f;
      else
         return 2;   // Allow minor variations with Linear and Cubic interpolations
   }

   bool CompareTolRelative() const
   {
      if (is_same<DataType, float>::value)
         return true;
      else
         return false;
   }

   void Create(uint Width, uint Height);

   void Free();

   ocipInterpolationType m_Interpolation;
   CImage<float> m_MapX, m_MapY;

   ocipImage m_CLMapX, m_CLMapY;

   void * m_NPPMapX;
   void * m_NPPMapY;
   int m_NPPMapXStep, m_NPPMapYStep;

   IPP_CODE(IppiRect m_IPPRemapROI;)

   IPP_CODE(int GetIPPMode() const
      {
         switch (m_Interpolation)
         {
         case ocipNearestNeighbour:
            return ippNearest;
         case ocipLinear:
            return ippLinear;
         case ocipCubic:
            return IPPI_INTER_CUBIC;
         case ocipSuperSampling:
            return IPPI_INTER_SUPER;
         default:
            return ippNearest;
         }
      } )

   NPP_CODE(NppiRect m_NPPRemapROI;)

   NPP_CODE(NppiInterpolationMode GetNPPMode() const
      {
         switch (m_Interpolation)
         {
         case ocipNearestNeighbour:
            return NPPI_INTER_NN;
         case ocipLinear:
            return NPPI_INTER_LINEAR;
         case ocipCubic:
            return NPPI_INTER_CUBIC;
         case ocipSuperSampling:
            return NPPI_INTER_SUPER;
         default:
            return NPPI_INTER_NN;
         }
      } )

   bool HasCVTest() const { return false; }
};

template<typename DataType>
class RemapLinearBench : public RemapBench<DataType>
{
public:
   RemapLinearBench()
   :  RemapBench<DataType>(ocipLinear)
   { }
};

template<typename DataType>
class RemapCubicBench : public RemapBench<DataType>
{
public:
   RemapCubicBench()
   :  RemapBench<DataType>(ocipCubic)
   { }
};

void GenerateXSinusPattern(CImage<float>& Img)
{
   double Amplitude = Img.Height / 15.;
   double Period = Img.Height / 4.;

   for (uint y = 0; y < Img.Height; y++)
      for (uint x = 0; x < Img.Width; x++)
      {
         double pos = x / Period;
         double val = sin(pos) * Amplitude;

         Img(x, y) = float(x + val);
      }

}

void GenerateYSinusPattern(CImage<float>& Img)
{
   double Amplitude = Img.Height / 15.;
   double Period = Img.Height / 6.;

   for (uint y = 0; y < Img.Height; y++)
      for (uint x = 0; x < Img.Width; x++)
      {
         double pos = y / Period;
         double val = sin(pos) * Amplitude;

         Img(x, y) = float(y + val);
      }

}

template<typename DataType>
void RemapBench<DataType>::Create(uint Width, uint Height)
{
   BenchUnaryBase<DataType>::Create(Width, Height);

   IPP_CODE(
      m_IPPRemapROI.x = 0;
      m_IPPRemapROI.y = 0;
      m_IPPRemapROI.width = Width;
      m_IPPRemapROI.height = Height;
      )

   NPP_CODE(
      m_NPPRemapROI.x = 0;
      m_NPPRemapROI.y = 0;
      m_NPPRemapROI.width = Width;
      m_NPPRemapROI.height = Height;
      )

   m_MapX.Create(Width, Height, 1, SImage::F32);
   m_MapY.Create(Width, Height, 1, SImage::F32);

   // Generate test pattern
   // A random map is an unrealistic test case so we generate the map based on sinusoidal data
   GenerateXSinusPattern(m_MapX);
   GenerateYSinusPattern(m_MapY);

   ocipCreateImage(&m_CLMapX, m_MapX, m_MapX.Data(), CL_MEM_READ_ONLY);
   ocipCreateImage(&m_CLMapY, m_MapY, m_MapY.Data(), CL_MEM_READ_ONLY);

   ocipSendImage(m_CLMapX);
   ocipSendImage(m_CLMapY);

   // NPP
   NPP_CODE(
      m_NPPMapX = NPP_Malloc<float>(Width, Height, m_NPPMapXStep, 1);
      m_NPPMapY = NPP_Malloc<float>(Width, Height, m_NPPMapYStep, 1);

      cudaMemcpy2D(m_NPPMapX, m_NPPMapXStep, m_MapX.Data(), m_MapX.Step,
         m_MapX.BytesWidth(), Height, cudaMemcpyHostToDevice);
      cudaMemcpy2D(m_NPPMapY, m_NPPMapYStep, m_MapY.Data(), m_MapY.Step,
         m_MapY.BytesWidth(), Height, cudaMemcpyHostToDevice);
      )
}

template<typename DataType>
void RemapBench<DataType>::Free()
{
   ocipReleaseImage(m_CLMapX);
   ocipReleaseImage(m_CLMapY);

   m_CLMapX = nullptr;
   m_CLMapY = nullptr;

   m_MapX.Free();
   m_MapY.Free();

   NPP_CODE(
      nppiFree(m_NPPMapX);
      nppiFree(m_NPPMapY);
      m_NPPMapX = nullptr;
      m_NPPMapY = nullptr;
   )

   BenchUnaryBase<DataType>::Free();
}

//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void RemapBench<DataType>::RunCL()
{
   ocipRemap(this->m_CLSrc, this->m_CLMapX, this->m_CLMapY, this->m_CLDst, this->m_Interpolation);
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void RemapBench<unsigned char>::RunIPP()
{ 
   IPP_CODE(
      ippiRemap_8u_C1R(this->m_ImgSrc.Data(), m_IPPRoi, this->m_ImgSrc.Step, this->m_IPPRemapROI,
         (Ipp32f*) this->m_MapX.Data(), this->m_MapX.Step, (Ipp32f*) this->m_MapY.Data(), this->m_MapY.Step,
         this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPRoi, GetIPPMode());
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void RemapBench<unsigned short>::RunIPP()
{
   IPP_CODE(
      ippiRemap_16u_C1R((Ipp16u*) this->m_ImgSrc.Data(), this->m_IPPRoi, this->m_ImgSrc.Step, this->m_IPPRemapROI,
         (Ipp32f*) this->m_MapX.Data(), this->m_MapX.Step, (Ipp32f*) this->m_MapY.Data(), this->m_MapY.Step,
         (Ipp16u*) this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPRoi, GetIPPMode());
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void RemapBench<unsigned char>::RunNPP()
{
   NPP_CODE(
      NppStatus status = nppiRemap_8u_C1R((Npp8u*) this->m_NPPSrc, this->m_NPPRoi, this->m_NPPSrcStep, this->m_NPPRemapROI,
         (Npp32f*) this->m_NPPMapX, this->m_NPPMapXStep, (Npp32f*) this->m_NPPMapY, this->m_NPPMapYStep,
         (Npp8u*) this->m_NPPDst, this->m_NPPDstStep, this->m_NPPRoi, GetNPPMode());
      status = status;
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void RemapBench<unsigned short>::RunNPP()
{
   NPP_CODE(
      nppiRemap_16u_C1R((Npp16u*) this->m_NPPSrc, this->m_NPPRoi, this->m_NPPSrcStep, this->m_NPPRemapROI,
         (Npp32f*) this->m_NPPMapX, this->m_NPPMapXStep, (Npp32f*) this->m_NPPMapY, this->m_NPPMapYStep,
         (Npp16u*) this->m_NPPDst, this->m_NPPDstStep, this->m_NPPRoi, GetNPPMode());
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void RemapBench<float>::RunNPP()
{
   NPP_CODE(
      nppiRemap_32f_C1R((Npp32f*) this->m_NPPSrc, this->m_NPPRoi, this->m_NPPSrcStep, this->m_NPPRemapROI,
         (Npp32f*) this->m_NPPMapX, this->m_NPPMapXStep, (Npp32f*) this->m_NPPMapY, this->m_NPPMapYStep,
         (Npp32f*) this->m_NPPDst, this->m_NPPDstStep, this->m_NPPRoi, GetNPPMode());
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void RemapBench<float>::RunIPP()
{
   IPP_CODE(
      ippiRemap_32f_C1R((Ipp32f*) this->m_ImgSrc.Data(), m_IPPRoi, this->m_ImgSrc.Step, this->m_IPPRemapROI,
         (Ipp32f*) this->m_MapX.Data(), this->m_MapX.Step, (Ipp32f*) this->m_MapY.Data(), this->m_MapY.Step,
         (Ipp32f*) this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPRoi, GetIPPMode());
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void RemapBench<DataType>::RunCV()
{
   // TODO
}
