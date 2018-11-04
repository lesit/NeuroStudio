#pragma once

#include <atlimage.h>

#include "../../common.h"
#include "../shape.h"
#include "../ImageProcessing.h"

namespace np
{
	namespace gui
	{
		namespace win32
		{
			class Win32Image
			{
			public:
				virtual ~Win32Image(){}

				CImage& GetImage(){ return m_img; }

				neuro_size_t GetDataSize() const {
					return m_data_size;
				}

				static NP_RECT GetRemoveBorderRect(const CImage& img, COLORREF bkcolor, neuro_u32 min_width_border = 1, neuro_u32 min_height_border = 1);

				const neuro_u32 m_channel;
				const neuro_u32 m_depth;
				const NP_SIZE m_cell_sz;

				const neuro_size_t m_data_size;
			protected:
				Win32Image(neuro_u32 channel, bool is_color, neuro_u32 width, neuro_u32 height);

				CImage m_img;
			};

			class ReadImage : public Win32Image
			{
				COLORREF m_bkcolor;
			public:
				ReadImage(bool is_color, neuro_u32 width, neuro_u32 height
					, neuro_float scale_min, neuro_float scale_max
					, COLORREF bkcolor = RGB(255, 255, 255));

				ReadImage(bool is_color, neuro_u32 width, neuro_u32 height, COLORREF bkcolor = RGB(255, 255, 255));

				ReadImage(bool is_color, neuro_u32 width, neuro_u32 height
					, const _IMAGEDATA_MONO_SCALE_INFO& mono_scale
					, neuro_float scale_min = -1.0f, neuro_float scale_max = 1.0f
					, COLORREF bkcolor = RGB(255, 255, 255));

				virtual ~ReadImage();

				void SetScaleInfo(neuro_float red_scale, neuro_float green_scale, neuro_float blue_scale
					, neuro_float scale_min = -1.0f, neuro_float scale_max = 1.0f);
				void SetScaleInfo(const _IMAGEDATA_MONO_SCALE_INFO& mono_scale
					, neuro_float scale_min = -1.0f, neuro_float scale_max = 1.0f);

				bool LoadImage(const wchar_t* path, _stretch_type stretch_type = _stretch_type::none);
				bool LoadImage(HBITMAP hBmp, _stretch_type stretch_type = _stretch_type::none);
				bool LoadImage(HBITMAP hBmp, const NP_RECT& rcBitmap, _stretch_type stretch_type = _stretch_type::none);

				bool ReadData(neuro_float* value, neuro_u32 size);

			protected:
				_IMAGEDATA_MONO_SCALE_INFO m_mono_scale;
				neuro_float m_scale_min;
				neuro_float m_scale_max;
			};

			class CreateImage : public Win32Image
			{
			public:
				CreateImage(neuro_u32 channel, bool is_color, neuro_u32 width, neuro_u32 height, bool isZeroBlack=false);
				virtual ~CreateImage();

				bool SetData(const neuro_float* value, neuro_u32 size);
				bool SetData(const neuro_float* value, neuro_u32 size, neuro_float scale_min, neuro_float scale_max);

				void Display(HDC dc, const NP_RECT& target_rc, NP_SIZE dst_cell, neuro_u32 cell_border = 1, COLORREF b_color=RGB(0,0,0));
				NP_RECT Display(HDC dc, const NP_RECT& rcShape, const _MAX_CELL_SIZE& max_cell_sz = _MAX_CELL_SIZE(), bool is_center=true, neuro_u32 cell_border = 1, COLORREF b_color = RGB(0, 0, 0));

			private:
				const bool m_isZeroBlack;
			};
		}
	}
}
