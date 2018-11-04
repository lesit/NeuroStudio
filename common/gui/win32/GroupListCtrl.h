#pragma once


// CGroupListCtrl
#define UM_GROUP_CLICK	WM_USER+1

namespace np
{
	namespace gui
	{
		namespace win32
		{
			class CGroupListCtrl : public CListCtrl
			{
				DECLARE_DYNAMIC(CGroupListCtrl)

			public:
				CGroupListCtrl();
				virtual ~CGroupListCtrl();

				LRESULT AddColumn(CString strHeader);

				LRESULT InsertGroupHeader(int nGroupId, CString strHeader, DWORD dwState = LVGS_NORMAL, DWORD dwAlign = LVGA_HEADER_LEFT);
				LRESULT InsertGroupItem(int nGroupId, int nIndex, LPCTSTR strItem, DWORD_PTR data=NULL);

				void ResizeHeader();

				int GetRowGroupId(int nRow);
			protected:
				DECLARE_MESSAGE_MAP()
				afx_msg void OnLButtonUp(UINT nFlags, CPoint point);
				virtual void PreSubclassWindow();

			protected:
				int GroupHitTest(const CPoint& point);

				BOOL SetGroupFooter(int nGroupID, CString footer, DWORD dwAlign = LVGA_FOOTER_CENTER);
				BOOL SetGroupSubtitle(int nGroupID, CString subtitle);
				BOOL SetGroupTitleImage(int nGroupID, int nImage, CString topDesc, CString bottomDesc);
			};

	/*	ListView에서 Group을 사용하려면 아래의 문구가 stdafx.h에 선언되어서 링크되어야 한다.
	#pragma comment(linker,"/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")
	*/

		}
	}
}
