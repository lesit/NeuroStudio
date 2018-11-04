#pragma once

#include "../shape.h"

namespace np
{
	namespace gui
	{
		namespace win32
		{
			enum class horz_align{ left, center, right };
			class TextDraw
			{
			public:
				static NP_SIZE CalculateTextSize(CDC& dc, const NP_RECT& drawArea, const std::wstring& str, bool bMultiLine = false)
				{
					if (str.size() <= 0)
						return NP_SIZE(0, 0);

					UINT uFlag = DT_CALCRECT;
					uFlag |= bMultiLine ? DT_WORDBREAK | DT_EDITCONTROL | DT_EXTERNALLEADING : DT_SINGLELINE;

					CRect rc(0, 0, drawArea.GetWidth(), drawArea.GetHeight());
					int height = 0;
					while ((height = dc.DrawText(str.c_str(), str.size(), rc, uFlag)))
					{
						if (rc.right > 0 && rc.bottom > 0)
							break;

						rc.bottom += 400;
					}

					return NP_SIZE(rc.Width(), rc.Height());
				}

				static void MultiText(CDC& dc, NP_RECT rcText, const std::wstring& str, horz_align align = horz_align::left, bool vert_center = true, NP_RECT* ret = NULL)
				{
					NP_SIZE sz = CalculateTextSize(dc, rcText, str, true);
					if (sz.width > rcText.GetWidth())
						sz.width = rcText.GetWidth();
					if (sz.height > rcText.GetHeight())
						sz.height = rcText.GetHeight();

					if (align == horz_align::center)
					{
						rcText.left += rcText.GetWidth() / 2 - sz.width / 2;
						rcText.right = rcText.left + sz.width;
					}

					if (vert_center)
						rcText.top += rcText.GetHeight() / 2 - sz.height / 2;
					rcText.bottom = rcText.top + sz.height;

					if (ret)
						*ret = rcText;

					CRect rc(rcText.left, rcText.top, rcText.right, rcText.bottom);
					dc.DrawText(str.c_str(), str.length(), rc, DT_WORDBREAK | DT_EDITCONTROL | DT_EXTERNALLEADING | DT_END_ELLIPSIS);
				}

				static void SingleText(CDC& dc, NP_RECT rcText, const std::wstring& str, horz_align align = horz_align::left, bool vert_center = true, bool revertEnd = false)
				{
					UINT flag = DT_SINGLELINE | DT_END_ELLIPSIS;
					if (vert_center)
						flag |= DT_VCENTER;

					if (revertEnd && align != horz_align::center)
					{
						NP_SIZE sz = CalculateTextSize(dc, rcText, str);
						if (sz.width > rcText.GetWidth())
						{
							if (align == horz_align::right)
							{
								align = horz_align::left;
								rcText.right = rcText.left + 500;
							}
							else if (align == horz_align::left)
							{
								align = horz_align::right;
								rcText.left = rcText.right - 500;
							}
						}
					}
					switch (align)
					{
					case horz_align::left:
						flag |= DT_LEFT;
						break;
					case horz_align::center:
						flag |= DT_CENTER;
						break;
					case horz_align::right:
						flag |= DT_RIGHT;
						break;
					}

					CRect rc(rcText.left, rcText.top, rcText.right, rcText.bottom);
					dc.DrawText(str.c_str(), str.length(), rc, flag);
				}
			};
		}
	}
}
