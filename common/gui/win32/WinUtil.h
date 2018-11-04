#pragma once

#include "../../common.h"

namespace np
{
	namespace gui
	{
		namespace win32
		{
			class WinUtil
			{
			public:
				static neuro_32 GetScrollInfo(neuro_32 nSBCode, neuro_32 nPage, neuro_32 nMax, neuro_32 nPos, neuro_u32 nTrackPos);
				static void ProcessScrollEvent(CWnd& wnd, int nType, neuro_32 nSBCode, neuro_u32 nTrackPos);

				static void AdjustListBoxHeight(CWnd& parent, CListBox& listBox);
				static void ResizeListBoxHScroll(CListBox& listbox);

				static void DeleteAllColumns(CListCtrl& listCtrl);
				static void ResizeListControlHeader(CListCtrl& listCtrl);
			};
		}
	}
}
