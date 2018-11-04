#include "stdafx.h"
#include "ScrollWnd.h"

#include "gui/Win32/WinUtil.h"

using namespace np::gui::win32;

CScrollWnd::CScrollWnd()
{
}

CScrollWnd::~CScrollWnd()
{
}

BOOL CScrollWnd::PreCreateWindow(CREATESTRUCT& cs)
{
	cs.style |= WS_HSCROLL | WS_VSCROLL;	// 요게 잘 안되면 OnCreate에서 하자

	return CWnd::PreCreateWindow(cs);
}

BEGIN_MESSAGE_MAP(CScrollWnd, CWnd)
	ON_WM_SIZE()
	ON_WM_HSCROLL()
	ON_WM_VSCROLL()
END_MESSAGE_MAP()

void CScrollWnd::RefreshDisplay(const NP_RECT* area)
{
	if (GetSafeHwnd() == NULL)
		return;

	if (area)
		InvalidateRect(CRect(area->left, area->top, area->right, area->bottom), FALSE);
	else
		Invalidate(FALSE);
}

void CScrollWnd::OnSize(UINT nType, int cx, int cy)
{
	RefreshScrollBars();
	OnScrollChanged();

	__super::OnSize(nType, cx, cy);
}

void CScrollWnd::OnHScroll(UINT nSBCode, UINT nTrackPos, CScrollBar* pScrollBar)
{
	gui::win32::WinUtil::ProcessScrollEvent(*this, SB_HORZ, nSBCode, nTrackPos);
	m_vpOrg = GetViewport();

	CWnd::OnHScroll(nSBCode, nTrackPos, pScrollBar);

	OnScrollChanged();
}

void CScrollWnd::OnVScroll(UINT nSBCode, UINT nTrackPos, CScrollBar* pScrollBar)
{
	gui::win32::WinUtil::ProcessScrollEvent(*this, SB_VERT, nSBCode, nTrackPos);
	m_vpOrg = GetViewport();

	CWnd::OnVScroll(nSBCode, nTrackPos, pScrollBar);

	OnScrollChanged();
}

NP_POINT CScrollWnd::GetViewport()
{
	NP_POINT ret;

	SCROLLINFO scInfo;
	GetScrollInfo(SB_HORZ, &scInfo);
	ret.x = scInfo.nPos * GetScrollMoving(true);

#ifdef _DEBUG
	if (scInfo.nPos > 0)
		int a = 0;
#endif;

	GetScrollInfo(SB_VERT, &scInfo);
	ret.y = scInfo.nPos * GetScrollMoving(false);
	return ret;
}

void CScrollWnd::SetViewport(const NP_POINT& org)
{
	SCROLLINFO scInfo;
	GetScrollInfo(SB_HORZ, &scInfo);
	scInfo.nPos = org.x / GetScrollMoving(true);
	SetScrollInfo(SB_HORZ, &scInfo);

	GetScrollInfo(SB_VERT, &scInfo);
	scInfo.nPos = org.y / GetScrollMoving(false);
	SetScrollInfo(SB_VERT, &scInfo);
}

void CScrollWnd::RefreshScrollBars()
{
	CRect clientRect;
	GetClientRect(&clientRect);
	if (!clientRect.IsRectEmpty())
	{
//		ShowScrollBar(SB_HORZ, FALSE);
//		ShowScrollBar(SB_VERT, FALSE);
	}

	auto refresh = [&](int type, int view_total_size, int moving, int track_pos)
	{
		int view_size = type == SB_HORZ ? clientRect.Width() : clientRect.Height();

//		int total = info.item_count * info.item_size / info.moving;
		int total = view_total_size / moving;
		int view = view_size / moving;

		if (total == 0 || total <= view)
		{
//			ShowScrollBar(type, FALSE);
			SCROLLINFO scInfo;
			memset(&scInfo, 0, sizeof(scInfo));
			scInfo.cbSize = sizeof(scInfo);
			scInfo.fMask = SIF_ALL;

			SetScrollInfo(type, &scInfo);
			return;
		}

		SCROLLINFO scInfo;
		memset(&scInfo, 0, sizeof(scInfo));
		GetScrollInfo(type, &scInfo);
		scInfo.cbSize = sizeof(scInfo);
		scInfo.fMask = SIF_ALL;
		scInfo.nMin = 0;
		scInfo.nMax = total - 1;
		scInfo.nPage = view;
		scInfo.nTrackPos = track_pos;

		SetScrollInfo(type, &scInfo);
//		ShowScrollBar(type, TRUE);
	};
	NP_SIZE total_view = GetScrollTotalViewSize();
	refresh(SB_HORZ, total_view.width, GetScrollMoving(true), GetScrollTrackPos(true));
	refresh(SB_VERT, total_view.height, GetScrollMoving(false), GetScrollTrackPos(false));
}
