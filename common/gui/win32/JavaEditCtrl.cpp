#include "stdafx.h"

#include "JavaEditCtrl.h"
#include "util/StringUtil.h"

using namespace np::gui::win32;

//// Java Edit Rich Edit Ctrl
CJavaEditCtrl::CJavaEditCtrl()
{
}

CJavaEditCtrl::~CJavaEditCtrl()
{
}

BEGIN_MESSAGE_MAP(CJavaEditCtrl, CRichEditCtrl)
	//{{AFX_MSG_MAP(CJavaEditCtrl)
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CJavaEditCtrl message handlers

void CJavaEditCtrl::PreSubclassWindow() 
{
	CHARFORMAT cf;

	cf.cbSize = sizeof (CHARFORMAT);  
	cf.dwMask = CFM_FACE | CFM_SIZE; 
	cf.yHeight = 180; 
	_stprintf_s(cf.szFaceName, _T("MS Sans Serif")); 
	SetDefaultCharFormat(cf); 
	SetOptions(ECOOP_OR, ECO_AUTOWORDSELECTION|ECO_AUTOVSCROLL|ECO_AUTOHSCROLL|ECO_WANTRETURN
		|ECO_NOHIDESEL|ECO_SAVESEL);
//	SetEventMask(GetEventMask() | ENM_KEYEVENTS);
	CRichEditCtrl::PreSubclassWindow();
}

BOOL CJavaEditCtrl::PreTranslateMessage(MSG* pMsg) 
{
	if(pMsg->message==WM_KEYDOWN)
	{
		long start;
		long end;
		if(!GetEditableArea(start, end))
			return TRUE;

		/// 쓸수 있는 영역을 벗어나면 수정을 못하게 한다.
		long st_sel, ed_sel;
		GetSel(st_sel, ed_sel);
		if(st_sel<start || st_sel>end || ed_sel<start || ed_sel>end)
		{
			SetSel(end, end);
			return TRUE;
		}

		if(pMsg->wParam==VK_TAB)
		{
			ReplaceSel(_T("\t"), TRUE);
			return TRUE;
		}
		else if(pMsg->wParam==VK_ESCAPE)
			return TRUE;
	}

	return CRichEditCtrl::PreTranslateMessage(pMsg);
}

bool CJavaEditCtrl::GetEditableArea(long &nSt, long &nEd)
{
	int nLine=GetLineCount();
	if(nLine<2)
		return false;

	nSt=LineIndex(1);
	nEd=LineIndex(nLine-1)-1;
	return true;
}

CString CJavaEditCtrl::GetBody()
{
	long nSt, nEd;
	if(!GetEditableArea(nSt, nEd))
		return _T("");
	
	CString str;
	GetWindowText(str);
	// nSt, nEd 는 cursor 단위이기 때문에 \r\n 을 \n로 바꾸어야 정확해진다.
	str.Replace(L"\r\n", L"\n");
	return str.Mid(nSt, nEd-nSt);
}

/// header만 바꾼다.
void CJavaEditCtrl::SetHeaderContent(CString strHeader)
{
	if(strHeader.IsEmpty())
	{
		SetWindowText(L"");
		return;
	}

	const COLORREF crFunction=RGB(0,127,127);
	const COLORREF crHeader=RGB(0,0,127);

	CHARFORMAT cf;
	cf.cbSize = sizeof (CHARFORMAT);  
	cf.dwMask		= CFM_COLOR | CFM_UNDERLINE | CFM_BOLD;
	cf.dwEffects	= (DWORD)~(CFE_AUTOCOLOR | CFE_UNDERLINE | CFE_BOLD);
	cf.crTextColor = crFunction;

//	HideSelection(TRUE, TRUE);
	if(GetLineCount()<3){	// 초기이면
		/// header와 end를 세팅하자.
		static wchar_t* empty_body=_T("{\n\n}");
		SetWindowText(strHeader+empty_body);
	}
	else
	{	// header만 살짝 바꾸어준다.
		long nEd=LineIndex(1)-1;
		SetSel(0,nEd);
		ReplaceSel(strHeader);
	}

	static TCHAR strFunction[]=_T("function");
	long nEd=_countof(strFunction);

	/// function부분의 색을 바꾸어준다.
	SetSel(0, nEd);
	SetSelectionCharFormat(cf);

	long nSt=nEd;
	nEd=nSt+strHeader.GetLength();
	SetSel(nSt, nEd);
	cf.crTextColor=crHeader;	// name 부분 부터 다른 색으로 하자
	SetSelectionCharFormat(cf);

	// { 색을 바꾸자
	cf.crTextColor=crFunction;
	SetSel(nEd, nEd+1);
	SetSelectionCharFormat(cf);

	// } 색을 바꾸자
	nEd=LineIndex(GetLineCount()-1);
	SetSel(nEd, nEd+1);
	SetSelectionCharFormat(cf);

	SetSel(nEd, nEd);
}

void CJavaEditCtrl::SetBodyContent(LPCTSTR strBody)
{
	long nSt, nEd;
	if(!GetEditableArea(nSt, nEd))
		return;

	CHARFORMAT cf;
	cf.cbSize = sizeof (CHARFORMAT);  
	cf.dwMask		= CFM_COLOR | CFM_UNDERLINE | CFM_BOLD;
	cf.dwEffects	= (DWORD)~(CFE_AUTOCOLOR | CFE_UNDERLINE | CFE_BOLD);
	cf.crTextColor = RGB(255,255,255);

	SetSel(nSt, nEd);
	ReplaceSel(strBody);
//	HideSelection(TRUE, TRUE);
	SetSel(nSt, nSt);
//	HideSelection(FALSE, TRUE);
//	HideSelection(FALSE, FALSE);

	SetSelectionCharFormat(cf);

	nEd=LineIndex(GetLineCount()-1);
	SetSel(nEd, nEd);
}
