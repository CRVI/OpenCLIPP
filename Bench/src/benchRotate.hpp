
template<typename DataType> class RotateBench;
template<typename DataType> class RotateLinearBench;

typedef RotateBench<unsigned char>   RotateBenchU8;
typedef RotateBench<unsigned short>  RotateBenchU16;
typedef RotateBench<float>           RotateBenchF32;

typedef RotateLinearBench<unsigned char>   RotateLinearBenchU8;
typedef RotateLinearBench<unsigned short>  RotateLinearBenchU16;
typedef RotateLinearBench<float>           RotateLinearBenchF32;

template<typename DataType>
class RotateBench : public BenchUnaryBase<DataType, false>
{
public:
   RotateBench(bool Interpolation = false)
   :  m_Interpolation(Interpolation)
   { }

   void RunIPP();
   void RunCL();
   void RunNPP();
   void RunCV();

   void Create(uint Width, uint Height)
   {
      BenchUnaryBase::Create(Width, Height);

      m_ImgDstIPP.MakeBlack();

      IPP_CODE(
         m_IPPRotROI.x = 0;
         m_IPPRotROI.y = 0;
         m_IPPRotROI.width = Width;
         m_IPPRotROI.height = Height;
         )

      m_Angle = 10;
      m_XShift = Width / 3;
      m_YShift = Height / 2;
   }

   bool m_Interpolation;
   double m_Angle;
   double m_XShift;
   double m_YShift;

   IPP_CODE(IppiRect m_IPPRotROI;)

   IPP_CODE(int IppInterpol()
      {
         int interpol_mode = IPPI_INTER_NN | IPPI_SMOOTH_EDGE;
         if (m_Interpolation)
            interpol_mode = IPPI_INTER_LINEAR | IPPI_SMOOTH_EDGE;
         return interpol_mode;
      } )

   NPP_CODE(NppiRect m_NPPRotROI;)

   NPP_CODE(int NppInterpol()
      {
         int interpol_mode = NPPI_INTER_NN;
         if (m_Interpolation)
            interpol_mode = NPPI_INTER_LINEAR;
         return interpol_mode;
      } )

   bool HasCVTest() const { return false; }
};

template<typename DataType>
class RotateLinearBench : public RotateBench<DataType>
{
public:
   RotateLinearBench()
   :  RotateBench(true)
   { }
};

//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void RotateBench<DataType>::RunCL()
{
   ocipRotate(m_CLSrc, m_CLDst, m_Angle, m_XShift, m_YShift, m_Interpolation);

   ocipReadImage(m_CLDst);
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void RotateBench<unsigned char>::RunIPP()
{ 
   IPP_CODE(
      ippiRotate_8u_C1R(this->m_ImgSrc.Data(), m_IPPRoi, this->m_ImgSrc.Step, m_IPPRotROI,
         this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, m_IPPRotROI, m_Angle, m_XShift, m_YShift, IppInterpol());
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void RotateBench<unsigned short>::RunIPP()
{
   IPP_CODE(
      ippiRotate_16u_C1R((Ipp16u*) this->m_ImgSrc.Data(), m_IPPRoi, this->m_ImgSrc.Step, m_IPPRotROI,
         (Ipp16u*) this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, m_IPPRotROI, m_Angle, m_XShift, m_YShift, IppInterpol());
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void RotateBench<unsigned char>::RunNPP()
{
   NPP_CODE(
      nppiRotate_8u_C1R((Npp8u*) this->m_NPPSrc, m_NPPRoi, this->m_NPPSrcStep, m_NPPRotROI,
         (Npp8u*) this->m_NPPDst, this->m_NPPDstStep, m_NPPRotROI, m_Angle, m_XShift, m_YShift, NppInterpol());
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void RotateBench<unsigned short>::RunNPP()
{
   NPP_CODE(
      nppiRotate_16u_C1R((Npp16u*) this->m_NPPSrc, m_NPPRoi, this->m_NPPSrcStep, m_NPPRotROI,
         (Npp16u*) this->m_NPPDst, this->m_NPPDstStep, m_NPPRotROI, m_Angle, m_XShift, m_YShift, NppInterpol());
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void RotateBench<float>::RunNPP()
{
   NPP_CODE(
      nppiRotate_32f_C1R((Npp32f*) this->m_NPPSrc, m_NPPRoi, this->m_NPPSrcStep, m_NPPRotROI,
         (Npp32f*) this->m_NPPDst, this->m_NPPDstStep, m_NPPRotROI, m_Angle, m_XShift, m_YShift, NppInterpol());
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void RotateBench<float>::RunIPP()
{
   IPP_CODE(
      ippiRotate_32f_C1R((Ipp32f*) this->m_ImgSrc.Data(), m_IPPRoi, this->m_ImgSrc.Step, m_IPPRotROI,
         (Ipp32f*) this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, m_IPPRotROI, m_Angle, m_XShift, m_YShift, IppInterpol());
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void RotateBench<DataType>::RunCV()
{
   // OpenCV OCL does not seem to have rotate
   //CV_CODE( rotate(m_CVSrc CONSTANT_MIDDLE CONSTANT_LAST , m_CVDst CV_PARAM_LAST ); )
}
