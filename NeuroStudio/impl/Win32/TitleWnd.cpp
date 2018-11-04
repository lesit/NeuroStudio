// TitleWnd.cpp : 구현 파일입니다.
//

#include "stdafx.h"
#include "TitleWnd.h"

#include "gui/Win32/GraphicUtil.h"

// CTitleWnd

IMPLEMENT_DYNAMIC(CTitleWnd, CWnd)

CTitleWnd::CTitleWnd()
{
	m_angle_degree = 0;
}

CTitleWnd::~CTitleWnd()
{
}

BEGIN_MESSAGE_MAP(CTitleWnd, CWnd)
	ON_WM_PAINT()
	ON_WM_ERASEBKGND()
END_MESSAGE_MAP()

bool CTitleWnd::Create(const wchar_t* title, UINT dwStyle, RECT rc, CWnd* parent, int angle_degree)
{
	if (!__super::Create(NULL, NULL, dwStyle, rc, parent, 0))
		return false;

	m_angle_degree = angle_degree;

	LOGFONT logFont;
	if (parent && parent->GetFont())
		memcpy(&logFont, parent->GetFont(), sizeof(LOGFONT));
	else
		memset(&logFont, 0, sizeof(LOGFONT));

	logFont.lfEscapement = m_angle_degree * 10;
	logFont.lfOrientation = m_angle_degree * 10;

	logFont.lfHeight = 15;
	logFont.lfWidth = logFont.lfHeight / 2;
	logFont.lfWeight = FW_BOLD;
	m_listTitleFont.CreateFontIndirect(&logFont);

	SetWindowText(title);
	return true;
}

#include "gui/win32/TextDraw.h"

void CTitleWnd::OnPaint()
{
	CPaintDC paintDC(this); // device context for painting

	CRect rcClient;
	GetClientRect(&rcClient);	// 전체 영역을 얻는다.

	CMemDC memDC(paintDC, rcClient);
	CDC& dc = memDC.GetDC();

	dc.FillSolidRect(&rcClient, RGB(90, 90, 90));

	dc.SetTextColor(RGB(255, 255, 255));

	CFont* oldFont = dc.SelectObject(&m_listTitleFont);

	CString str;
	GetWindowText(str);
	NP_SIZE sz = gui::win32::TextDraw::CalculateTextSize(dc, NP_RECT(rcClient.left, rcClient.top, rcClient.right, rcClient.bottom), (const wchar_t*)str);

	int x, y;
	if (m_angle_degree == 0)
	{
		x = rcClient.Width() / 2 - sz.width / 2;
		y = rcClient.Height() / 2 - sz.height / 2;
	}
	else if (m_angle_degree == 90)
	{
		x = rcClient.Width() / 2 + sz.height / 2;
		y = rcClient.Height() / 2 + sz.width / 2;
	}
	else if (m_angle_degree == 180)
	{
		x = rcClient.Width() / 2 + sz.width / 2;
		y = rcClient.Height() / 2 + sz.height / 2;
	}
	else
	{
		x = rcClient.Width() / 2 + sz.height / 2;
		y = rcClient.Height() / 2 - sz.width / 2;
	}
	dc.TextOut(x, y, str);

	dc.SelectObject(oldFont);
}


BOOL CTitleWnd::OnEraseBkgnd(CDC* pDC)
{
	// TODO: Add your message handler code here and/or call default

	return FALSE;
}
