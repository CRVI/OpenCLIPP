////////////////////////////////////////////////////////////////////////////////
//! @file	: Blob.cl
//! @date   : Jul 2013
//!
//! @brief  : Blob labeling
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


constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;


#ifdef I

   // For signed integer images
   #define READ_IMAGE(img, pos) convert_uint4(read_imagei(img, sampler, pos))

#else // I

   #ifdef UI

      // For unsigned integer images
      #define READ_IMAGE(img, pos) read_imageui(img, sampler, pos)

   #else // UI

      // For float
      #define READ_IMAGE(img, pos) convert_uint4(read_imagef(img, sampler, pos))

   #endif // UI

#endif // I


struct SBlobInfo
{
   int ConnectType;	// 4 or 8
   int NbBlobs;
   int LastUsefulIteration;
};

#define LABEL_BACKGROUND 0xffffffff

uint GetLabel(int gx, int gy)
{
   return gy << 16 | gx;
}

int2 GetPosFromLabel(uint Label)
{
   int2 pos;
   pos.x = Label & 0xFFFF;
   pos.y = (Label >> 16) & 0xFFFF;
   return pos;
}

uint FindMasterLabel(global uint * equivTable, uint table_step, uint Label)
{
   int2 Pos = GetPosFromLabel(Label);
   
   uint Parent = equivTable[Pos.y * table_step + Pos.x];

   while (Label != Parent)
   {
      Label = Parent;
      Pos = GetPosFromLabel(Label);
      Parent = equivTable[Pos.y * table_step + Pos.x];
   }

   return Label;
}

uint GetNeighboursMin(global const uint * labelImg, int gx, int gy, int img_width, int img_height, uint ConnectType, uint label_step, uint Label)
{
   int NbNeighbours = 0;
   uint Neighbours[8];

   if (gx > 0)
      Neighbours[NbNeighbours++] = gy * label_step + gx - 1;

   if (gx < img_width - 1)
      Neighbours[NbNeighbours++] = gy * label_step + gx + 1;

   if (gy > 0)
   {
      Neighbours[NbNeighbours++] = (gy - 1) * label_step + gx;

      if (ConnectType == 8)
      {
         if (gx > 0)
            Neighbours[NbNeighbours++] = (gy - 1) * label_step + gx - 1;

         if (gx < img_width - 1)
            Neighbours[NbNeighbours++] = (gy - 1) * label_step + gx + 1;
      }

   }

   if (gy < img_height - 1)
   {
      Neighbours[NbNeighbours++] = (gy + 1) * label_step + gx;

      if (ConnectType == 8)
      {
         if (gx > 0)
            Neighbours[NbNeighbours++] = (gy + 1) * label_step + gx - 1;

         if (gx < img_width - 1)
            Neighbours[NbNeighbours++] = (gy + 1) * label_step + gx + 1;
      }

   }

   uint NeighboursMin = Label;

   for (int i = 0; i < NbNeighbours; i++)
      NeighboursMin = min(NeighboursMin, labelImg[Neighbours[i]]);

   return NeighboursMin;
}

kernel void init_label(read_only image2d_t source,
   global uint * labelImg, global uint * equivTable, 
   uint label_step, uint table_step,
   global struct SBlobInfo * info)
{
   const int gx = get_global_id(0);
   const int gy = get_global_id(1);

   const bool master = (gx == 0 && gy == 0);

   const int2 pos = {gx, gy};

   label_step /= sizeof(uint);	// step is in bytes but we want it in pixel
   table_step /= sizeof(uint);	// step is in bytes but we want it in pixel

   const uint label_index = gy * label_step + gx;
   const uint table_index = gy * table_step + gx;

   uint4 color = READ_IMAGE(source, pos);

   bool ValidPixel = color.x > 0;

   // Initialize label image
   if (ValidPixel)
   {
      // Set the initial label for the current pixel
      uint Label = GetLabel(gx, gy);
      labelImg[label_index] = Label;
      equivTable[table_index] = Label;
   }
   else
   {
      labelImg[label_index] = LABEL_BACKGROUND;
      equivTable[table_index] = LABEL_BACKGROUND;
   }

}

kernel void label_step1(global uint * labelImg, global uint * equivTable, 
   uint label_step, uint table_step,
   global struct SBlobInfo * info, int Iter)
{
   const int img_width = get_global_size(0);
   const int img_height = get_global_size(0);
   const int gx = get_global_id(0);
   const int gy = get_global_id(1);

   label_step /= sizeof(uint);	// step is in bytes but we want it in pixel
   table_step /= sizeof(uint);	// step is in bytes but we want it in pixel

   const uint label_index = gy * label_step + gx;

   uint Label = labelImg[label_index];

   bool ValidPixel = Label != LABEL_BACKGROUND;

   if (ValidPixel)
   {
      uint NeighboursMin = GetNeighboursMin(labelImg, gx, gy, img_width, img_height, info->ConnectType, label_step, Label);

      if (NeighboursMin < Label)
      {
         // We found a lower value
         // Set the equivalence table of current label to the new value
         int2 pos = GetPosFromLabel(Label);
         uint OldValue = atomic_min(&equivTable[pos.y * table_step + pos.x], NeighboursMin);
         if (NeighboursMin < OldValue)
            info->LastUsefulIteration = Iter;
      }

   }
   
}

kernel void label_step2(global uint * labelImg, global uint * equivTable, 
   uint label_step, uint table_step,
   global struct SBlobInfo * info, int Iter)
{
   if (info->LastUsefulIteration < Iter)
      return;

   const int gx = get_global_id(0);
   const int gy = get_global_id(1);

   label_step /= sizeof(uint);	// step is in bytes but we want it in pixel
   table_step /= sizeof(uint);	// step is in bytes but we want it in pixel

   const uint label_index = gy * label_step + gx;
   const uint table_index = gy * table_step + gx;

   uint Label = labelImg[label_index];

   bool ValidPixel = Label != LABEL_BACKGROUND;

   if (ValidPixel)
   {
      uint MasterLabel = FindMasterLabel(equivTable, table_step, Label);

      if (Label != MasterLabel)
      {
         equivTable[table_index] = MasterLabel;
         labelImg[label_index] = MasterLabel;
      }

   }
 
}


kernel void reorder_labels1(global uint * labelImg, global uint * tmpBuffer, 
   uint label_step, uint table_step,
   global struct SBlobInfo * info)
{
   const int gx = get_global_id(0);
   const int gy = get_global_id(1);

   label_step /= sizeof(uint);	// step is in bytes but we want it in pixel
   table_step /= sizeof(uint);	// step is in bytes but we want it in pixel

   const uint label_index = gy * label_step + gx;
   const uint table_index = gy * table_step + gx;

   uint Label = labelImg[label_index];

   if (Label != LABEL_BACKGROUND)
   {
      if (Label == GetLabel(gx, gy))
      {
         // This is the root of the label
         uint NewLabel = atomic_inc(&info->NbBlobs);
         tmpBuffer[table_index] = NewLabel;
      }

   }

}

kernel void reorder_labels2(global uint * labelImg, global uint * tmpBuffer, 
   uint label_step, uint table_step,
   global struct SBlobInfo * info)
{
   const int gx = get_global_id(0);
   const int gy = get_global_id(1);

   label_step /= sizeof(uint);	// step is in bytes but we want it in pixel
   table_step /= sizeof(uint);	// step is in bytes but we want it in pixel

   const uint label_index = gy * label_step + gx;

   uint Label = labelImg[label_index];

   if (Label != LABEL_BACKGROUND)
   {
      int2 Pos = GetPosFromLabel(Label);
      labelImg[label_index] = tmpBuffer[Pos.y * table_step + Pos.x];
   }

}
