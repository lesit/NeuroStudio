#pragma once

#include "TextDraw.h"
#include "../../tensor/tensor_shape.h"
#include "../../util/StringUtil.h"
namespace np
{
	namespace gui
	{
		namespace win32
		{
			class ListDraw
			{
			public:
				static std::wstring GetFormat(bool has_index, neuro_u16 decimal_place_right)
				{
					std::wstring format;
					if (has_index)
						format += L"%u : ";
					format += util::StringUtil::Format<wchar_t>(L"%%+.%uf", neuro_u32(decimal_place_right));
					return format;
				}

				inline static std::wstring GetValueString(std::wstring& format, bool has_index, neuro_size_t i, neuro_float value)
				{
					std::wstring str;
					if (has_index)
						str = np::util::StringUtil::Format<wchar_t>(format.c_str(), i, value);
					else
						str = np::util::StringUtil::Format<wchar_t>(format.c_str(), value);
					return str;
				}

				static const neuro_u32 horz_gap = 5;
				static const neuro_u32 vert_gap = 3;
				static NP_SIZE GetDrawValueSize(CDC& dc, const bool has_index, neuro_u16 decimal_place_right)
				{
					std::wstring format = GetFormat(has_index, decimal_place_right);

					NP_SIZE sz = TextDraw::CalculateTextSize(dc, NP_RECT(0,0,1024,1024), GetValueString(format, has_index, 0, 11111111.11111111f));
					sz.width += horz_gap * 2;
					sz.height += vert_gap * 2;

					return sz;
				}

				static NP_SIZE GetDrawSize(CDC& dc, const bool has_index, neuro_u16 decimal_place_right, neuro_size_t value_size)
				{
					NP_SIZE sz = GetDrawValueSize(dc, has_index, decimal_place_right);
					sz.height *= value_size;
					return sz;
				}

				static NP_SIZE Draw(CDC& dc, const NP_RECT& rc, const bool has_index, neuro_u16 decimal_place_right
					, const neuron_value* value, neuro_size_t size)
				{
					if (decimal_place_right > 8)
						decimal_place_right = 8;

					std::wstring format = GetFormat(has_index, decimal_place_right);

					CMemDC memDC(dc, CRect(rc.left, rc.top, rc.right, rc.bottom));
					CDC& listDC = memDC.GetDC();
					listDC.FillSolidRect(CRect(rc.left, rc.top, rc.right, rc.bottom), RGB(255, 255, 255));
					listDC.SetBkMode(OPAQUE);

					listDC.SelectObject(dc.GetCurrentFont());

					NP_SIZE sz = GetDrawValueSize(dc, has_index, decimal_place_right);

					NP_RECT rcItem(rc.left, rc.top, rc.left + sz.width, rc.top + sz.height);
					for (neuro_size_t i = 0; i < size; i++, value++)
					{
						listDC.Rectangle(rcItem.left, rcItem.top, rcItem.right, rcItem.bottom+1);

						NP_RECT rcText = rcItem;
						rcText.left += horz_gap;
						rcText.right -= horz_gap;
						TextDraw::SingleText(listDC, rcText, GetValueString(format, has_index, i, *value));

						if (rcItem.bottom>rc.bottom - vert_gap)
							break;

						rcItem.top = rcItem.bottom;
						rcItem.bottom = rcItem.top + sz.height;
					}

					sz.height *= size;
					return sz;
				}
			};
		}
	}
}
