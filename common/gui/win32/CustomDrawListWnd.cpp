#include "stdafx.h"
#include "CustomDrawListWnd.h"

#include "gui/Win32/WinUtil.h"

using namespace np::gui::win32;

CCustomDrawListWnd::CCustomDrawListWnd()
{
	m_bkcolor = RGB(255, 255, 255);
	m_line_color = RGB(128, 0, 0);
}

CCustomDrawListWnd::~CCustomDrawListWnd()
{
}

BOOL CCustomDrawListWnd::PreCreateWindow(CREATESTRUCT& cs)
{
	cs.style |= WS_HSCROLL | WS_VSCROLL;	// 요게 잘 안되면 OnCreate에서 하자

	return CWnd::PreCreateWindow(cs);
}

BEGIN_MESSAGE_MAP(CCustomDrawListWnd, CWnd)
	ON_WM_ERASEBKGND()
	ON_WM_PAINT()
	ON_WM_SIZE()
	ON_WM_HSCROLL()
	ON_WM_VSCROLL()
	ON_WM_LBUTTONUP()
END_MESSAGE_MAP()

BOOL CCustomDrawListWnd::OnEraseBkgnd(CDC* pDC)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.

	return FALSE;
}

void CCustomDrawListWnd::OnPaint()
{
	CRect clientRect;
	GetClientRect(&clientRect);

	if (clientRect.left>clientRect.right || clientRect.top>clientRect.bottom)
		return;

	CPaintDC paintDC(this); // device context for painting
	CMemDC memDC(paintDC, clientRect);

	CDC& dc = memDC.GetDC();
	//	dc.SetBkMode(TRANSPARENT);
	dc.SetBkMode(OPAQUE);

	CBrush bkBrush;bkBrush.CreateSolidBrush(m_bkcolor);
	CBrush* pOldbrush = dc.SelectObject(&bkBrush);

	dc.FillSolidRect(clientRect, m_bkcolor);

	RefreshScrollBars();

	/*
	CPen line_pen(PS_DOT, 1, m_line_color);
	CPen* pOldPen = dc.SelectObject(&line_pen);

	dc.Rectangle(clientRect);
	*/
	SCROLLINFO scInfo;
	GetScrollInfo(SB_HORZ, &scInfo);
	const int item_org_left=scInfo.nPos;

	GetScrollInfo(SB_VERT, &scInfo);
	int start_item=scInfo.nPos;

	CDC itemDC;itemDC.CreateCompatibleDC(&dc);

	// Select a compatible bitmap into the memory DC
	int item_height = GetItemHeight();
	CRect ItemRect(0, 0, GetMaxItemWidth(), item_height);
	CBitmap bitmap; bitmap.CreateCompatibleBitmap(&paintDC, ItemRect.Width(), item_height);
	itemDC.SelectObject(&bitmap);

	LOGFONT logfont;
	memset(&logfont, 0, sizeof(LOGFONT));
	logfont.lfHeight = 14;
//	logfont.lfWidth = logfont.lfHeight / 2;

	CFont font;
	font.CreateFontIndirect(&logfont);
	CFont* pOldFont = itemDC.SelectObject(&font);

	int draw_top = 0;
	const int draw_width = clientRect.right - item_org_left;

	itemDC.FillSolidRect(ItemRect, m_bkcolor);

	long header_height = GetHeaderHeight();
	if (header_height > 0)
	{
		DrawHeader(itemDC, ItemRect);
		dc.BitBlt(0, draw_top, draw_width, item_height, &itemDC, item_org_left, 0, SRCCOPY);
		draw_top += header_height;
	}

	neuro_32 end = start_item + GetViewItemCount(clientRect.Height() - draw_top);
	if (end > GetItemCount())
		end = GetItemCount();

	for (neuro_32 item = start_item; item < end; item++)
	{
		itemDC.FillSolidRect(ItemRect, m_bkcolor);
		DrawItem(item, itemDC, ItemRect);

		if (draw_top + item_height>clientRect.bottom)
			item_height = clientRect.bottom - draw_top;

		dc.BitBlt(0, draw_top, draw_width, item_height, &itemDC, item_org_left, 0, SRCCOPY);
		draw_top += item_height;
	}
	itemDC.SelectObject(pOldFont);

//	dc.SelectObject(pOldPen);
	dc.SelectObject(pOldbrush);
}

neuro_u32 CCustomDrawListWnd::GetViewItemCount(neuro_u32 height) const
{
	return NP_Util::CalculateCountPer(height, GetItemHeight());
}

neuro_u32 CCustomDrawListWnd::GetMaxItemWidth() const
{
	neuro_u32 max_width = GetItemWidth();
	if (max_width == 0)	// 고정 너비가 아닌 경우
	{
		for (size_t i = 0, n = GetItemCount(); i < n; i++)
		{
			neuro_u32 width = GetItemWidth(i);
			if (width>max_width)
				max_width = width;
		}
	}
	return max_width;
}

void CCustomDrawListWnd::OnSize(UINT nType, int cx, int cy)
{
	RefreshScrollBars();

	__super::OnSize(nType, cx, cy);
}

void CCustomDrawListWnd::RefreshScrollBars()
{
	CRect clientRect;
	GetClientRect(&clientRect);
	if (!clientRect.IsRectEmpty())
		int a = 0;

	long header_height = GetHeaderHeight();
	if (header_height > 0)
		clientRect.top += header_height;

	RefreshScrollBar(SB_VERT, GetItemCount(), clientRect.Height()/ GetItemHeight());

	RefreshScrollBar(SB_HORZ, GetMaxItemWidth(), clientRect.Width());
}

void CCustomDrawListWnd::RefreshScrollBar(int nType, int total, int view)
{
	if (total==0 || total <= view)
	{
		ShowScrollBar(nType, FALSE);
		return;
	}

	SCROLLINFO scInfo;
	memset(&scInfo, 0, sizeof(scInfo));
	GetScrollInfo(nType, &scInfo);
	scInfo.cbSize = sizeof(scInfo);
	scInfo.fMask = SIF_ALL;
	scInfo.nMin = 0;
	scInfo.nMax = total - 1;
	scInfo.nPage = view;
	scInfo.nTrackPos = 1;

	SetScrollInfo(nType, &scInfo);
	ShowScrollBar(nType, TRUE);
}

void CCustomDrawListWnd::OnHScroll(UINT nSBCode, UINT nTrackPos, CScrollBar* pScrollBar)
{
	gui::win32::WinUtil::ProcessScrollEvent(*this, SB_HORZ, nSBCode, nTrackPos);

	CWnd::OnHScroll(nSBCode, nTrackPos, pScrollBar);
}

void CCustomDrawListWnd::OnVScroll(UINT nSBCode, UINT nTrackPos, CScrollBar* pScrollBar)
{
	gui::win32::WinUtil::ProcessScrollEvent(*this, SB_VERT, nSBCode, nTrackPos);

	CWnd::OnVScroll(nSBCode, nTrackPos, pScrollBar);
}


void CCustomDrawListWnd::OnLButtonUp(UINT nFlags, CPoint point)
{
	CRect clientRect;
	GetClientRect(&clientRect);

	SCROLLINFO scInfo;
	GetScrollInfo(SB_HORZ, &scInfo);
	int item_org_left = scInfo.nPos;

	GetScrollInfo(SB_VERT, &scInfo);
	neuro_32 start_item = scInfo.nPos;

	// Select a compatible bitmap into the memory DC
	int item_height = GetItemHeight();

	int draw_top = 0;
	const int draw_width = clientRect.right - item_org_left;

	long header_height = GetHeaderHeight();
	if (header_height > 0)
		draw_top += header_height + 5;

	if (point.y >= draw_top && point.y <= clientRect.Height())
	{
		neuro_32 iItem = start_item + (point.y - draw_top) / item_height;

		OnItemLButtonUp(iItem, point.x + item_org_left, point.y-draw_top);
	}

	CWnd::OnLButtonUp(nFlags, point);
}
