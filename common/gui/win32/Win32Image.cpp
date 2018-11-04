#include "stdafx.h"

#include "Win32Image.h"
#include "util/np_util.h"

using namespace np::gui;
using namespace np::gui::win32;

Win32Image::Win32Image(neuro_u32 channel, bool is_color, neuro_u32 width, neuro_u32 height)
: m_channel(channel), m_depth(is_color ? 3 : 1), m_cell_sz(width, height)
, m_data_size(m_channel * m_depth * neuro_u32(width) * neuro_u32(height))
{
}

NP_RECT Win32Image::GetRemoveBorderRect(const CImage& img, COLORREF bkcolor, neuro_u32 min_width_border, neuro_u32 min_height_border)
{
	if (img.IsNull())
		return NP_RECT();

	int width = img.GetWidth();
	int height = img.GetHeight();
	int nPitch = img.GetPitch();

	int bpp = img.GetBPP();
	if (bpp != 24)	// 24bit 로 저정했으니까 당연히!!
		return NP_RECT();

	NP_RECT ret(0, 0, width, height);

	const neuro_u8* bits = (const neuro_u8*)img.GetBits();
	const neuro_u8* bit_ptr = bits;
	// remove top border
	for (long y = ret.top; y < ret.bottom; y++)
	{
		long x = ret.left;
		for (; x < ret.right; x++)
		{
			const neuro_u8* pixel = bit_ptr + ((x*bpp) / 8);
			if (RGB(pixel[2], pixel[1], pixel[0]) != bkcolor)
			{
				ret.top = y;
				break;
			}
		}

		bit_ptr += nPitch;

		if (x != ret.right)
			break;
	}
	bit_ptr = bits + (height-1)*nPitch;
	for (long y = ret.bottom - 1; y > ret.top; y--)
	{
		long x = ret.left;
		for (; x < ret.right; x++)
		{
			const neuro_u8* pixel = bit_ptr + ((x*bpp) / 8);
			if (RGB(pixel[2], pixel[1], pixel[0]) != bkcolor)
			{
				ret.bottom = y;
				break;
			}
		}

		bit_ptr -= nPitch;

		if (x != ret.right)
			break;
	}

	for (long x = ret.left; x < ret.right; x++)
	{
		long y = ret.top;
		for (; y < ret.bottom; y++)
		{
			const neuro_u8* t = &bits[y*nPitch + x] + ((x*bpp) / 8);
			const neuro_u8* pixel = bits + y*nPitch + ((x*bpp) / 8);
			if (RGB(pixel[2], pixel[1], pixel[0]) != bkcolor)
			{
				ret.left = x;
				break;
			}
		}
		if (y != ret.bottom)
			break;
	}
	for (long x = ret.right - 1; x > ret.left; x--)
	{
		long y = ret.top;
		for (; y < ret.bottom; y++)
		{
			const neuro_u8* pixel = bits + y*nPitch + ((x*bpp) / 8);
			if (RGB(pixel[2], pixel[1], pixel[0]) != bkcolor)
			{
				ret.right = x;
				break;
			}
		}
		if (y != ret.bottom)
			break;
	}

	if (min_width_border >= width / 2)
		min_width_border = 0;
	if (min_height_border >= height / 2)
		min_height_border = 0;
	if (ret.top > min_height_border)
		ret.top -= min_height_border;
	if (ret.bottom < height - 1 - min_height_border)
		ret.bottom += min_height_border;
	if (ret.left > min_width_border)
		ret.left -= min_width_border;
	if (ret.right < width - 1 - min_width_border)
		ret.right += min_width_border;
	return ret;
}

ReadImage::ReadImage(bool is_color, neuro_u32 width, neuro_u32 height
	, const _IMAGEDATA_MONO_SCALE_INFO& mono_scale, neuro_float scale_min, neuro_float scale_max
	, COLORREF bkcolor)
	: Win32Image(1, is_color, width, height)
{
	m_bkcolor = bkcolor;

	m_img.Create(m_cell_sz.width, m_cell_sz.height, 24);

	memcpy(&m_mono_scale, &mono_scale, sizeof(_IMAGEDATA_MONO_SCALE_INFO));
	m_scale_min = -1.0f;
	m_scale_max = 1.0f;
}

ReadImage::ReadImage(bool is_color, neuro_u32 width, neuro_u32 height, COLORREF bkcolor)
: ReadImage(is_color, width, height, _IMAGEDATA_MONO_SCALE_INFO(), -1.0, 1.0, bkcolor)
{
}

ReadImage::ReadImage(bool is_color, neuro_u32 width, neuro_u32 height
	, neuro_float scale_min, neuro_float scale_max, COLORREF bkcolor)
	: ReadImage(is_color, width, height, _IMAGEDATA_MONO_SCALE_INFO(), scale_min, scale_max, bkcolor)
{
}

ReadImage::~ReadImage()
{
}

void ReadImage::SetScaleInfo(neuro_float red_scale, neuro_float green_scale, neuro_float blue_scale
	, neuro_float scale_min, neuro_float scale_max)
{
	m_mono_scale.red_scale = red_scale;
	m_mono_scale.green_scale = green_scale;
	m_mono_scale.blue_scale = blue_scale;
	m_scale_min = scale_min;
	m_scale_max = scale_max;
}

void ReadImage::SetScaleInfo(const _IMAGEDATA_MONO_SCALE_INFO& mono_scale, neuro_float scale_min, neuro_float scale_max)
{
	memcpy(&m_mono_scale, &mono_scale, sizeof(_IMAGEDATA_MONO_SCALE_INFO));

	m_scale_min = scale_min;
	m_scale_max = scale_max;
}

bool ReadImage::LoadImage(const wchar_t* path, _stretch_type stretch_type)
{
	// scale the original image to image definition size
	Gdiplus::Bitmap orgImg(path);
	if (orgImg.GetLastStatus() != Gdiplus::Ok)
		return false;

	HBITMAP	hBmp = NULL;
	bool bRet = orgImg.GetHBITMAP(NULL, &hBmp) == Gdiplus::Ok;

	if (bRet)
		bRet = LoadImage(hBmp, stretch_type);

	::DeleteObject(hBmp);

	return bRet;
}

bool ReadImage::LoadImage(HBITMAP hBmp, _stretch_type stretch_type)
{
	BITMAP bitmap;
	if (::GetObject(hBmp, sizeof(BITMAP), &bitmap) != sizeof(BITMAP))
		return false;

	return LoadImage(hBmp, NP_RECT(0, 0, bitmap.bmWidth, bitmap.bmHeight), stretch_type);
}

bool ReadImage::LoadImage(HBITMAP hBmp, const NP_RECT& rcBitmap, _stretch_type stretch_type)
{
	NP_SIZE source_sz = rcBitmap.GetSize();

	HDC src_dc = CreateCompatibleDC(NULL);
	HBITMAP hOld = (HBITMAP)SelectObject(src_dc, hBmp);

	HDC dst_dc = m_img.GetDC();
	::SetBkColor(dst_dc, m_bkcolor);
	CRect rc(0, 0, m_img.GetWidth(), m_img.GetHeight());
	::ExtTextOut(dst_dc, 0, 0, ETO_OPAQUE, &rc, NULL, 0, NULL);

	if (stretch_type == _stretch_type::none || source_sz == m_cell_sz)
	{
		::BitBlt(dst_dc, 0, 0, m_cell_sz.width, m_cell_sz.height, src_dc, 0, 0, SRCCOPY);
	}
	else
	{
		NP_SIZE target_sz = np::GetRatioShape(source_sz, m_cell_sz, stretch_type);

		NP_RECT stretch_ret = np::GetCenterShape(target_sz, m_cell_sz);
		SetStretchBltMode(dst_dc, HALFTONE);
		::StretchBlt(dst_dc, stretch_ret.left, stretch_ret.top, stretch_ret.GetWidth(), stretch_ret.GetHeight()
			, src_dc, rcBitmap.left, rcBitmap.top, rcBitmap.GetWidth(), rcBitmap.GetHeight(), SRCCOPY);
	}

	m_img.ReleaseDC();

	SelectObject(src_dc, hOld);
	DeleteDC(src_dc);
	return true;
}

bool ReadImage::ReadData(neuro_float* value, neuro_u32 size)
{
	int width = m_img.GetWidth();
	int height = m_img.GetHeight();
	int nPitch = m_img.GetPitch();

	int bpp = m_img.GetBPP();
	if (bpp != 24)	// 24bit 로 저정했으니까 당연히!!
		return false;

	if (size != width*height*m_depth)
		return false;

	const neuro_float scale_trans = (m_scale_max - m_scale_min) / neuro_float(255);
	const neuro_u8* bits = (const neuro_u8*)m_img.GetBits();
	if (m_depth == 1)
	{
		neuro_float* ptr = value;

		// x, y를 unsigned로 하면 아래 pixel 포인터 계산할때, 보통 -인 nPitch 가 값이 이상해진다. 따라서 integer로 해야한다.
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				const neuro_u8* pixel = bits + ((x*bpp) / 8);

				*ptr = neuro_float(pixel[2])*m_mono_scale.red_scale + neuro_float(pixel[1]) * m_mono_scale.green_scale + neuro_float(pixel[0]) * m_mono_scale.blue_scale;
				*ptr = *ptr * scale_trans + m_scale_min;

				++ptr;
			}

			bits += nPitch;
		}
//		DEBUG_OUTPUT(L"-->");
//		np::NP_Util::DebugOutputValues(value, size, width);
//		DEBUG_OUTPUT(L"<--");
	}
	else
	{
		const int area_size = width*height;

		neuro_float* r_value = value;
		neuro_float* g_value = r_value + area_size;
		neuro_float* b_value = g_value + area_size;
		// x, y를 unsigned로 하면 아래 pixel 포인터 계산할때, 보통 -인 nPitch 가 값이 이상해진다. 따라서 integer로 해야한다.
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				const neuro_u8* pixel = bits + ((x*bpp) / 8);

				*b_value = neuro_float(pixel[0]) * scale_trans + m_scale_min;
				*g_value = neuro_float(pixel[1]) * scale_trans + m_scale_min;
				*r_value = neuro_float(pixel[2]) * scale_trans + m_scale_min;

				++b_value;
				++g_value;
				++r_value;
			}
			bits += nPitch;
		}
	}
	return true;
}

CreateImage::CreateImage(neuro_u32 channel, bool is_color, neuro_u32 width, neuro_u32 height, bool isZeroBlack)
	: Win32Image(channel, is_color, width, height), m_isZeroBlack(isZeroBlack)
{
	m_img.Create(m_channel*m_cell_sz.width, m_cell_sz.height, 24);
}

CreateImage::~CreateImage()
{
}

bool CreateImage::SetData(const neuro_float* value, neuro_u32 size)
{
	neuro_float scale_min, scale_max;
	GetMinMax(value, size, scale_min, scale_max);

	return SetData(value, size, scale_min, scale_max);
}

bool CreateImage::SetData(const neuro_float* value, neuro_u32 size, neuro_float scale_min, neuro_float scale_max)
{
	if (size != m_data_size)
		return false;

	int nPitch = m_img.GetPitch();

	int bpp = m_img.GetBPP();
	if (bpp != 24)	// 24bit 로 저정했으니까 당연히!!
		return false;

	if (scale_min >= scale_max)
		GetMinMax(value, size, scale_min, scale_max);

	neuro_float scale_trans;
	if (scale_max == scale_min)
		scale_trans = 1;
	else
		scale_trans = 255 / (scale_max - scale_min);

	bpp /= 8;

	neuro_u8* upper_border_bits = (neuro_u8*)m_img.GetBits();
	neuro_u8* lower_border_bits = upper_border_bits + m_img.GetHeight()*nPitch;

	for (neuro_u32 d = 0; d < m_channel; d++)
	{
		neuro_u8* bits = (neuro_u8*)m_img.GetBits();
		const int left = d*m_cell_sz.width;

		for (int y = 0; y < m_cell_sz.height; y++)
		{
			// left border
			int x = left;

			for (int i = 0; i < m_cell_sz.width; i++, x++)
			{
				neuro_u8* pixel = bits + x*bpp;
				if (m_depth == 1)
				{
					pixel[2] = neuro_u8((*value - scale_min) * scale_trans);
					if (!m_isZeroBlack)
						pixel[2] = 255 - pixel[2];

					pixel[0] = pixel[1] = pixel[2];
					++value;
				}
				else
				{
					for (neuro_u32 i = 0; i < 3; i++, ++value)
					{
						pixel[i] = neuro_u8((*value - scale_min) * scale_trans);
						if (!m_isZeroBlack)
							pixel[i] = 255 - pixel[i];
					}
				}
			}

			bits += nPitch;
		}
	}
	return true;
}

void CreateImage::Display(HDC dc, const NP_RECT& rc, NP_SIZE dst_cell, neuro_u32 cell_border, COLORREF b_color)
{
	COLORREF old_bk = ::SetBkColor(dc, b_color);
	::SetBkMode(dc, OPAQUE);

	bool is_stretch = false;
	if (dst_cell != m_cell_sz)
	{
		is_stretch = true;
		SetStretchBltMode(dc, HALFTONE);
	}

	auto draw = [&]()
	{
		int src_x = 0;
		int last_x = m_img.GetWidth();
		for (int target_y = rc.top + cell_border; target_y + dst_cell.height + cell_border <= rc.bottom; target_y += dst_cell.height + cell_border)
		{
			for (int target_x = rc.left + cell_border; target_x + dst_cell.width + cell_border <= rc.right; target_x += dst_cell.width + cell_border)
			{
				CRect draw_rc(target_x - cell_border, target_y - cell_border, target_x + dst_cell.width + cell_border,
					target_y + dst_cell.height + cell_border);
				::ExtTextOut(dc, 0, 0, ETO_OPAQUE, &draw_rc, NULL, 0, NULL);

				if (is_stretch)
					m_img.StretchBlt(dc, target_x, target_y, dst_cell.width, dst_cell.height
						, src_x, 0, m_cell_sz.width, m_cell_sz.height, SRCCOPY);
				else
					m_img.BitBlt(dc, target_x, target_y, dst_cell.width, dst_cell.height, src_x, 0, SRCCOPY);

				src_x += m_cell_sz.width;
				if (src_x >= last_x)
					return;
			}
		}
	};
	draw();
	SetBkColor(dc, old_bk);
}

NP_RECT CreateImage::Display(HDC dc, const NP_RECT& rc, const _MAX_CELL_SIZE& max_cell_sz, bool is_center, neuro_u32 cell_border, COLORREF b_color)
{
#ifdef _DEBUG
	if (rc.left == 355)
		int a = 0;
#endif

	NP_RECT target_rc;
	NP_SIZE dst_cell;
	{
		NP_SIZE dst_rc_sz;
		GetDisplaySize(m_channel, m_cell_sz, rc.GetSize(), max_cell_sz, cell_border, dst_rc_sz, dst_cell);

		if (is_center)
		{
			target_rc.left = rc.left + (rc.GetWidth() - dst_rc_sz.width) / 2;
			target_rc.right = target_rc.left + dst_rc_sz.width;

			target_rc.top = rc.top + (rc.GetHeight() - dst_rc_sz.height) / 2;
			target_rc.bottom = target_rc.top + dst_rc_sz.height;
		}
		else
		{
			target_rc = NP_RECT({ rc.left, rc.top }, dst_rc_sz);
		}
	}

	Display(dc, target_rc, dst_cell);
	return target_rc;
}
