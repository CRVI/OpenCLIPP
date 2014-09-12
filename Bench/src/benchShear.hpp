
template<typename DataType> class ShearBench;
template<typename DataType> class ShearLinearBench;
template<typename DataType> class ShearCubicBench;


template<typename DataType>
class ShearBench : public BenchUnaryBase<DataType>
{
public:
   ShearBench(ocipInterpolationType Interpolation = ocipNearestNeighbour)
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

      m_ShearX = -0.5;
      m_ShearY = 1.2;
      m_XShift = Width / 5.;
      m_YShift = Height / -2.;
   }

   ocipInterpolationType m_Interpolation;
   double m_ShearX;
   double m_ShearY;
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
class ShearLinearBench : public ShearBench<DataType>
{
public:
   ShearLinearBench()
   :  ShearBench<DataType>(ocipLinear)
   { }
};

template<typename DataType>
class ShearCubicBench : public ShearBench<DataType>
{
public:
   ShearCubicBench()
   :  ShearBench<DataType>(ocipCubic)
   { }
};


//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void ShearBench<DataType>::RunCL()
{
   ocipShear(this->m_CLSrc, this->m_CLDst, this->m_ShearX, this->m_ShearY, this->m_XShift, this->m_YShift, this->m_Interpolation);
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void ShearBench<unsigned char>::RunIPP()
{ 
   IPP_CODE(
      ippiShear_8u_C1R(this->m_ImgSrc.Data(), m_IPPRoi, this->m_ImgSrc.Step, this->m_IPPRotROI,
         this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPRotROI, this->m_ShearX, this->m_ShearY, this->m_XShift, this->m_YShift, GetIPPMode());
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void ShearBench<unsigned short>::RunIPP()
{
   IPP_CODE(
      ippiShear_16u_C1R((Ipp16u*) this->m_ImgSrc.Data(), this->m_IPPRoi, this->m_ImgSrc.Step, this->m_IPPRotROI,
         (Ipp16u*) this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPRotROI, this->m_ShearX, this->m_ShearY, this->m_XShift, this->m_YShift, GetIPPMode());
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void ShearBench<unsigned char>::RunNPP()
{
   NPP_CODE(
      nppiShear_8u_C1R((Npp8u*) this->m_NPPSrc, this->m_NPPRoi, this->m_NPPSrcStep, this->m_NPPRotROI,
         (Npp8u*) this->m_NPPDst, this->m_NPPDstStep, this->m_NPPRotROI, this->m_ShearX, this->m_ShearY, this->m_XShift, this->m_YShift, GetNPPMode());
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void ShearBench<unsigned short>::RunNPP()
{
   NPP_CODE(
      nppiShear_16u_C1R((Npp16u*) this->m_NPPSrc, this->m_NPPRoi, this->m_NPPSrcStep, this->m_NPPRotROI,
         (Npp16u*) this->m_NPPDst, this->m_NPPDstStep, this->m_NPPRotROI, this->m_ShearX, this->m_ShearY, this->m_XShift, this->m_YShift, GetNPPMode());
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void ShearBench<float>::RunNPP()
{
   NPP_CODE(
      nppiShear_32f_C1R((Npp32f*) this->m_NPPSrc, this->m_NPPRoi, this->m_NPPSrcStep, this->m_NPPRotROI,
         (Npp32f*) this->m_NPPDst, this->m_NPPDstStep, this->m_NPPRotROI, this->m_ShearX, this->m_ShearY, this->m_XShift, this->m_YShift, GetNPPMode());
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void ShearBench<float>::RunIPP()
{
   IPP_CODE(
      ippiShear_32f_C1R((Ipp32f*) this->m_ImgSrc.Data(), m_IPPRoi, this->m_ImgSrc.Step, this->m_IPPRotROI,
         (Ipp32f*) this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPRotROI, this->m_ShearX, this->m_ShearY, this-> m_XShift, this->m_YShift, GetIPPMode());
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void ShearBench<DataType>::RunCV()
{
   // TODO
}
