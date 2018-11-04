// GroupListCtrl.cpp : 구현 파일입니다.
//

#include "stdafx.h"
#include "GroupListCtrl.h"
#include <shlwapi.h>

// CGroupListCtrl
using namespace np::gui::win32;

namespace WindowControl
{
	LRESULT EnableWindowTheme(HWND hwnd, LPCWSTR classList, LPCWSTR subApp, LPCWSTR idlist)
	{
		LRESULT lResult = S_FALSE;
	
		HRESULT (__stdcall *pSetWindowTheme)(HWND hwnd, LPCWSTR pszSubAppName, LPCWSTR pszSubIdList);
		HANDLE (__stdcall *pOpenThemeData)(HWND hwnd, LPCWSTR pszClassList);
		HRESULT (__stdcall *pCloseThemeData)(HANDLE hTheme);

		HMODULE hinstDll = ::LoadLibrary(_T("UxTheme.dll"));
		if(hinstDll)
		{
			(FARPROC&)pOpenThemeData = ::GetProcAddress(hinstDll, "OpenThemeData");
			(FARPROC&)pCloseThemeData = ::GetProcAddress(hinstDll, "CloseThemeData");
			(FARPROC&)pSetWindowTheme = ::GetProcAddress(hinstDll, "SetWindowTheme");
			if(pSetWindowTheme && pOpenThemeData && pCloseThemeData)
			{
				HANDLE theme = pOpenThemeData(hwnd,classList);
				if(theme!=NULL)
				{
					VERIFY(pCloseThemeData(theme)==S_OK);
					lResult = pSetWindowTheme(hwnd, subApp, idlist);
				}
			}
			::FreeLibrary(hinstDll);
		}
		return lResult;
	}

	bool IsCommonControlsEnabled()
	{
		bool commoncontrols = false;
	
		// Test if application has access to common controls
		HMODULE hinstDll = ::LoadLibrary(_T("comctl32.dll"));
		if(hinstDll)
		{
			DLLGETVERSIONPROC pDllGetVersion = (DLLGETVERSIONPROC)::GetProcAddress(hinstDll, "DllGetVersion");
			if(pDllGetVersion != NULL)
			{
				DLLVERSIONINFO dvi = {0};
				dvi.cbSize = sizeof(dvi);
				HRESULT hRes = pDllGetVersion ((DLLVERSIONINFO *) &dvi);
				if(SUCCEEDED(hRes))
					commoncontrols = dvi.dwMajorVersion >= 6;
			}
			::FreeLibrary(hinstDll);
		}
		return commoncontrols;
	}

	bool IsThemeEnabled()
	{
		bool XPStyle = false;
		bool (__stdcall *pIsAppThemed)();
		bool (__stdcall *pIsThemeActive)();

		// Test if operating system has themes enabled
		HMODULE hinstDll = ::LoadLibrary(_T("UxTheme.dll"));
		if(hinstDll)
		{
			(FARPROC&)pIsAppThemed = ::GetProcAddress(hinstDll, "IsAppThemed");
			(FARPROC&)pIsThemeActive = ::GetProcAddress(hinstDll,"IsThemeActive");
			if(pIsAppThemed != NULL && pIsThemeActive != NULL)
			{
				if(pIsAppThemed() && pIsThemeActive())
				{
					// Test if application has themes enabled by loading the proper DLL
					XPStyle = IsCommonControlsEnabled();
				}
			}
			::FreeLibrary(hinstDll);
		}
		return XPStyle;
	}
}

IMPLEMENT_DYNAMIC(CGroupListCtrl, CListCtrl)

CGroupListCtrl::CGroupListCtrl()
{
}

CGroupListCtrl::~CGroupListCtrl()
{
}


BEGIN_MESSAGE_MAP(CGroupListCtrl, CListCtrl)
	ON_WM_LBUTTONUP()
END_MESSAGE_MAP()

void CGroupListCtrl::ResizeHeader()
{
	CHeaderCtrl* pHeaderCtrl = GetHeaderCtrl();
	int nColumn=pHeaderCtrl->GetItemCount();
	for(int i=0;i<nColumn;i++){
		SetColumnWidth(i, LVSCW_AUTOSIZE_USEHEADER);
	}
}

void CGroupListCtrl::PreSubclassWindow()
{
	CListCtrl::PreSubclassWindow();

	SetWindowLong(GetSafeHwnd(), GWL_STYLE, WS_CHILD | WS_VISIBLE | WS_BORDER |
		LVS_REPORT | LVS_SINGLESEL | LVS_SHOWSELALWAYS );

	// Focus retangle is not painted properly without double-buffering
	DWORD extStyle=GetExtendedStyle();
	extStyle|=LVS_EX_DOUBLEBUFFER;
	extStyle|=LVS_EX_FULLROWSELECT|LVS_EX_GRIDLINES;
	SetExtendedStyle(extStyle);

	// Enable Vista-look if possible
	WindowControl::EnableWindowTheme(GetSafeHwnd(), L"ListView", L"Explorer", NULL);

	EnableGroupView(TRUE);
}

LRESULT CGroupListCtrl::AddColumn(CString strHeader)
{
	return InsertColumn(GetHeaderCtrl()->GetItemCount(), strHeader);
}

LRESULT CGroupListCtrl::InsertGroupHeader(int nGroupId, CString strHeader, DWORD dwState /* = LVGS_NORMAL */, DWORD dwAlign /*= LVGA_HEADER_LEFT*/)
{
	LVGROUP lg = {0};
	lg.cbSize = sizeof(lg);
	lg.iGroupId = nGroupId;
	lg.state = LVGS_NORMAL;
	lg.mask = LVGF_GROUPID | LVGF_HEADER | LVGF_STATE | LVGF_ALIGN;
	lg.uAlign = dwAlign;

	lg.pszHeader = (LPWSTR)(LPCTSTR)strHeader;
	lg.cchHeader = strHeader.GetLength();

	return InsertGroup(nGroupId, (PLVGROUP)&lg );
}

LRESULT CGroupListCtrl::InsertGroupItem(int nGroupId, int nIndex, LPCTSTR strText, DWORD_PTR data)
{
	LVITEM lvItem = {0};
	lvItem.mask = LVIF_GROUPID | LVIF_TEXT | LVIF_PARAM;
	lvItem.iItem = nIndex;
	lvItem.iSubItem = 0;
	lvItem.iGroupId = nGroupId;
	lvItem.pszText=(LPWSTR)(LPCTSTR)strText;
	lvItem.lParam = data;
	return InsertItem(&lvItem);
}

int CGroupListCtrl::GetRowGroupId(int nRow)
{
	LVITEM lvi = {0};
	lvi.mask = LVIF_GROUPID;
	lvi.iItem = nRow;
	VERIFY( GetItem(&lvi) );
	return lvi.iGroupId;
}

int CGroupListCtrl::GroupHitTest(const CPoint& point)
{
	if(HitTest(point)!=-1)
		return -1;

	// We require that each group contains atleast one item
	if(GetItemCount()==0)
		return -1;

	// This logic doesn't support collapsible groups
	int nFirstRow = -1;
	CRect gridRect;
	GetWindowRect(&gridRect);
	for(CPoint pt = point ; pt.y < gridRect.bottom ; pt.y += 2)
	{
		nFirstRow = HitTest(pt);
		if(nFirstRow!=-1)
			break;
	}

	if(nFirstRow==-1)
		return -1;

	int nGroupId = GetRowGroupId(nFirstRow);

	// Extra validation that the above row belongs to a different group
	int nAboveRow = GetNextItem(nFirstRow,LVNI_ABOVE);
	if(nAboveRow!=-1 && nGroupId==GetRowGroupId(nAboveRow))
		return -1;

	return nGroupId;
}

BOOL CGroupListCtrl::SetGroupFooter(int nGroupID, CString footer, DWORD dwAlign /*= LVGA_FOOTER_CENTER*/)
{
	LVGROUP lg = {0};
	lg.cbSize = sizeof(lg);
	lg.mask = LVGF_FOOTER | LVGF_ALIGN;
	lg.uAlign = dwAlign;
	lg.pszFooter = (LPWSTR)(LPCTSTR)footer;
	lg.cchFooter = footer.GetLength();

	if(SetGroupInfo(nGroupID, (PLVGROUP)&lg)==-1)
		return FALSE;

	return TRUE;
}

BOOL CGroupListCtrl::SetGroupSubtitle(int nGroupID, CString subtitle)
{
	LVGROUP lg = {0};
	lg.cbSize = sizeof(lg);
	lg.mask = LVGF_SUBTITLE;
	lg.pszSubtitle = (LPWSTR)(LPCTSTR)subtitle;
	lg.cchSubtitle = subtitle.GetLength();

	if(SetGroupInfo(nGroupID, (PLVGROUP)&lg)==-1)
		return FALSE;

	return TRUE;
}

BOOL CGroupListCtrl::SetGroupTitleImage(int nGroupID, int nImage, CString topDesc, CString bottomDesc)
{
	LVGROUP lg = {0};
	lg.cbSize = sizeof(lg);
	lg.mask = LVGF_TITLEIMAGE;
	lg.iTitleImage = nImage;	// Index of the title image in the control imagelist.

	if(!topDesc.IsEmpty())
	{
		// Top description is drawn opposite the title image when there is
		// a title image, no extended image, and uAlign==LVGA_HEADER_CENTER.
		lg.mask |= LVGF_DESCRIPTIONTOP;
		lg.pszDescriptionTop = (LPWSTR)(LPCTSTR)topDesc;
		lg.cchDescriptionTop = topDesc.GetLength();
	}
	if(!bottomDesc.IsEmpty())
	{
		// Bottom description is drawn under the top description text when there is
		// a title image, no extended image, and uAlign==LVGA_HEADER_CENTER.
		lg.mask |= LVGF_DESCRIPTIONBOTTOM;
		lg.pszDescriptionBottom = (LPWSTR)(LPCTSTR)bottomDesc;
		lg.cchDescriptionBottom = bottomDesc.GetLength();
	}

	if(SetGroupInfo(nGroupID, (PLVGROUP)&lg)==-1)
		return FALSE;

	return TRUE;
}

void CGroupListCtrl::OnLButtonUp(UINT nFlags, CPoint point)
{
	CListCtrl::OnLButtonUp(nFlags, point);

	int iGroupId = GroupHitTest(point);
	if(iGroupId>=0)
	{
		CWnd* pParent=GetParent();
		if(pParent)
			pParent->SendMessage(UM_GROUP_CLICK, 0, iGroupId);
	}
}
