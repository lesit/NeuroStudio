#include "stdafx.h"

#include "DragPropertyCtrl.h"

using namespace ahnn::windows;

/////////////////////////////////////////////////////////////////////////////
CDragPropertyCtrl::CDragPropertyCtrl()
{
	m_pLastProp = NULL;
}

CDragPropertyCtrl::~CDragPropertyCtrl()
{
}

BEGIN_MESSAGE_MAP(CDragPropertyCtrl, CMFCPropertyGridCtrl)
END_MESSAGE_MAP()

void CDragPropertyCtrl::PreSubclassWindow()
{
	__super::PreSubclassWindow();

	ASSERT(::IsWindow(m_hWnd));
	AfxMakeDragList(m_hWnd);
}

BOOL CDragPropertyCtrl::BeginDrag(CPoint pt)
{
	m_pLastProp = NULL;
	DrawInsert(HitTest(pt));
	return TRUE;
}

void CDragPropertyCtrl::CancelDrag(CPoint)
{
	DrawInsert(NULL);
}

UINT CDragPropertyCtrl::Dragging(CPoint pt)
{
	CMFCPropertyGridProperty* pProp=HitTest(pt);
	DrawInsert(pProp);
	return (pProp == NULL) ? DL_STOPCURSOR : DL_MOVECURSOR;
}

void CDragPropertyCtrl::Dropped(CPoint pt)
{
	ASSERT(!(GetStyle() & (LBS_OWNERDRAWFIXED|LBS_OWNERDRAWVARIABLE)) ||
		(GetStyle() & LBS_HASSTRINGS));

	DrawInsert(NULL);

	CMFCPropertyGridProperty* pSrcProp=GetCurSel();
	if(!pSrcProp)
		return;

	POSITION prevPos=NULL;
	POSITION srcPos=m_lstProps.Find(pSrcProp);
	POSITION dstPos=NULL;
	for (POSITION pos = m_lstProps.GetHeadPosition(); pos != NULL;)
	{
		POSITION curPos=pos;

		CMFCPropertyGridProperty* pProp = m_lstProps.GetNext(pos);
		ASSERT_VALID(pProp);

		CMFCPropertyGridProperty* pDstProp=pProp->HitTest(pt);
		if (pDstProp != NULL)
		{
			if(pDstProp==pSrcProp || prevPos==srcPos)
				break;

			dstPos=curPos;
			break;
		}

		prevPos=curPos;
	}

	if(!dstPos)
		return;

	m_lstProps.InsertBefore(dstPos, pSrcProp);
	m_lstProps.RemoveAt(srcPos);
	ReposProperties();

	RedrawWindow();
}

void CDragPropertyCtrl::DrawInsert(CMFCPropertyGridProperty* pProp)
{
	if (m_pLastProp != pProp)
	{
		DrawSingle(m_pLastProp);
		DrawSingle(pProp);
		m_pLastProp = pProp;
	}
}

void CDragPropertyCtrl::DrawSingle(CMFCPropertyGridProperty* pProp)
{
	if (pProp == NULL)
		return;

	CBrush* pBrush = CDC::GetHalftoneBrush();
	CRect rect;
	GetClientRect(&rect);
	CRgn rgn;
	rgn.CreateRectRgnIndirect(&rect);

	CDC* pDC = GetDC();
	// prevent drawing outside of listbox
	// this can happen at the top of the listbox since the listbox's DC is the
	// parent's DC
	pDC->SelectClipRgn(&rgn);

	rect=pProp->GetRect();
	rect.bottom = rect.top+2;
	rect.top -= 2;
	CBrush* pBrushOld = pDC->SelectObject(pBrush);
	//draw main line
	pDC->PatBlt(rect.left, rect.top, rect.Width(), rect.Height(), PATINVERT);

	pDC->SelectObject(pBrushOld);
	ReleaseDC(pDC);
}

BOOL CDragPropertyCtrl::OnChildNotify(UINT nMessage, WPARAM wParam, LPARAM lParam, LRESULT* pResult)
{
	if (nMessage != m_nMsgDragList)
		return __super::OnChildNotify(nMessage, wParam, lParam, pResult);

	ASSERT(pResult != NULL);
	LPDRAGLISTINFO pInfo = (LPDRAGLISTINFO)lParam;
	ASSERT(pInfo != NULL);
	CPoint pt=pInfo->ptCursor;
	ScreenToClient(&pt);
	switch (pInfo->uNotification)
	{
	case DL_BEGINDRAG:
		*pResult = BeginDrag(pt);
		break;
	case DL_CANCELDRAG:
		CancelDrag(pt);
		break;
	case DL_DRAGGING:
		*pResult = Dragging(pt);
		break;
	case DL_DROPPED:
		Dropped(pt);
		break;
	}
	return TRUE;
}
