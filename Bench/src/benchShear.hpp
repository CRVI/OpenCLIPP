
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
   void RunCV();

   bool HasNPPTest() const { return false; }    // NPP does not implement Shear primitive

   float CompareTolerance() const { return 0.01f; }
   bool CompareTolRelative() const { return true; }

   void Create(uint Width, uint Height)
   {
      BenchUnaryBase<DataType>::Create(Width, Height);

      IPP_CODE(
         m_ImgDstIPP.MakeBlack();
         m_IPPShearROI.x = 0;
         m_IPPShearROI.y = 0;
         m_IPPShearROI.width = Width;
         m_IPPShearROI.height = Height;
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

   IPP_CODE(IppiRect m_IPPShearROI;)

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
      ippiShear_8u_C1R(this->m_ImgSrc.Data(), m_IPPRoi, this->m_ImgSrc.Step, this->m_IPPShearROI,
         this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPShearROI, this->m_ShearX, this->m_ShearY, this->m_XShift, this->m_YShift, GetIPPMode());
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void ShearBench<unsigned short>::RunIPP()
{
   IPP_CODE(
      ippiShear_16u_C1R((Ipp16u*) this->m_ImgSrc.Data(), this->m_IPPRoi, this->m_ImgSrc.Step, this->m_IPPShearROI,
         (Ipp16u*) this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPShearROI, this->m_ShearX, this->m_ShearY, this->m_XShift, this->m_YShift, GetIPPMode());
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void ShearBench<float>::RunIPP()
{
   IPP_CODE(
      ippiShear_32f_C1R((Ipp32f*) this->m_ImgSrc.Data(), m_IPPRoi, this->m_ImgSrc.Step, this->m_IPPShearROI,
         (Ipp32f*) this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPShearROI, this->m_ShearX, this->m_ShearY, this-> m_XShift, this->m_YShift, GetIPPMode());
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void ShearBench<DataType>::RunCV()
{
   // TODO
}
