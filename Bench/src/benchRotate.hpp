
template<typename DataType> class RotateBench;
template<typename DataType> class RotateLinearBench;
template<typename DataType> class RotateCubicBench;


template<typename DataType>
class RotateBench : public BenchUnaryBase<DataType>
{
public:
   RotateBench(ocipInterpolationType Interpolation = ocipNearestNeighbour)
   :  m_Interpolation(Interpolation)
   { }

   void RunIPP();
   void RunCL();
   void RunNPP();
   void RunCV();

   float CompareTolerance() const { return 0.01f; }
   bool CompareTolRelative() const { return true; }

   void Create(uint Width, uint Height)
   {
      BenchUnaryBase<DataType>::Create(Width, Height);

      IPP_CODE(
         m_ImgDstIPP.MakeBlack();
         m_IPPRotROI.x = 0;
         m_IPPRotROI.y = 0;
         m_IPPRotROI.width = Width;
         m_IPPRotROI.height = Height;
         )

      NPP_CODE(
         m_ImgDstNPP.MakeBlack();
         cudaMemcpy2D(m_NPPDst, m_NPPDstStep, m_ImgDstNPP.Data(), m_ImgDstNPP.Step,
            m_ImgDstNPP.BytesWidth(), Height, cudaMemcpyHostToDevice);
         m_NPPRotROI.x = 0;
         m_NPPRotROI.y = 0;
         m_NPPRotROI.width = Width;
         m_NPPRotROI.height = Height;
         )

      m_Angle = 10;
      m_XShift = Width / 3;
      m_YShift = Height / 2;
   }

   ocipInterpolationType m_Interpolation;
   double m_Angle;
   double m_XShift;
   double m_YShift;

   IPP_CODE(IppiRect m_IPPRotROI;)

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

   NPP_CODE(NppiRect m_NPPRotROI;)

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
class RotateLinearBench : public RotateBench<DataType>
{
public:
   RotateLinearBench()
   :  RotateBench<DataType>(ocipLinear)
   { }
};

template<typename DataType>
class RotateCubicBench : public RotateBench<DataType>
{
public:
   RotateCubicBench()
   :  RotateBench<DataType>(ocipCubic)
   { }
};


//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void RotateBench<DataType>::RunCL()
{
   ocipRotate(this->m_CLBufferSrc, this->m_CLBufferDst, this->m_Angle, this->m_XShift, this->m_YShift, this->m_Interpolation);
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void RotateBench<unsigned char>::RunIPP()
{ 
   IPP_CODE(
      ippiRotate_8u_C1R(this->m_ImgSrc.Data(), m_IPPRoi, this->m_ImgSrc.Step, this->m_IPPRotROI,
         this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPRotROI, this->m_Angle, this->m_XShift, this->m_YShift, GetIPPMode());
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void RotateBench<unsigned short>::RunIPP()
{
   IPP_CODE(
      ippiRotate_16u_C1R((Ipp16u*) this->m_ImgSrc.Data(), this->m_IPPRoi, this->m_ImgSrc.Step, this->m_IPPRotROI,
         (Ipp16u*) this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPRotROI, this->m_Angle, this->m_XShift, this->m_YShift, GetIPPMode());
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void RotateBench<unsigned char>::RunNPP()
{
   NPP_CODE(
      nppiRotate_8u_C1R((Npp8u*) this->m_NPPSrc, this->m_NPPRoi, this->m_NPPSrcStep, this->m_NPPRotROI,
         (Npp8u*) this->m_NPPDst, this->m_NPPDstStep, this->m_NPPRotROI, this->m_Angle, this->m_XShift, this->m_YShift, GetNPPMode());
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void RotateBench<unsigned short>::RunNPP()
{
   NPP_CODE(
      nppiRotate_16u_C1R((Npp16u*) this->m_NPPSrc, this->m_NPPRoi, this->m_NPPSrcStep, this->m_NPPRotROI,
         (Npp16u*) this->m_NPPDst, this->m_NPPDstStep, this->m_NPPRotROI, this->m_Angle, this->m_XShift, this->m_YShift, GetNPPMode());
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void RotateBench<float>::RunNPP()
{
   NPP_CODE(
      nppiRotate_32f_C1R((Npp32f*) this->m_NPPSrc, this->m_NPPRoi, this->m_NPPSrcStep, this->m_NPPRotROI,
         (Npp32f*) this->m_NPPDst, this->m_NPPDstStep, this->m_NPPRotROI, this->m_Angle, this->m_XShift, this->m_YShift, GetNPPMode());
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void RotateBench<float>::RunIPP()
{
   IPP_CODE(
      ippiRotate_32f_C1R((Ipp32f*) this->m_ImgSrc.Data(), m_IPPRoi, this->m_ImgSrc.Step, this->m_IPPRotROI,
         (Ipp32f*) this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPRotROI, this->m_Angle,this-> m_XShift, this->m_YShift, GetIPPMode());
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void RotateBench<DataType>::RunCV()
{
   // OpenCV OCL does not seem to have rotate
   //CV_CODE( rotate(this->m_CVSrc CONSTANT_MIDDLE CONSTANT_LAST , this->m_CVDst CV_PARAM_LAST ); )
}
