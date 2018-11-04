#include "stdafx.h"

#include "WinUtil.h"

using namespace np::gui::win32;

neuro_32 WinUtil::GetScrollInfo(neuro_32 nSBCode, neuro_32 nPage, neuro_32 nMax, neuro_32 nPos, neuro_u32 nTrackPos)
{
	{
		neuro_32 nNewPos = nPos;
		switch (nSBCode)
		{
		case SB_LINELEFT:	// SB_LINEUP
//			TRACE(L"scroll. SB_LINELEFT\r\n");
			nNewPos -= 1;
			break;
		case SB_PAGELEFT:
//			TRACE(L"scroll. SB_PAGELEFT\r\n");
			nNewPos -= nPage;
			break;
		case SB_LINERIGHT:
//			TRACE(L"scroll. SB_LINERIGHT\r\n");
			nNewPos += 1;
			break;
		case SB_PAGERIGHT:
//			TRACE(L"scroll. SB_PAGERIGHT\r\n");
			nNewPos += nPage;
			break;
		case SB_ENDSCROLL:
//			TRACE(L"scroll. SB_ENDSCROLL\r\n");
			break;
		case SB_THUMBTRACK:
//			TRACE(L"scroll. SB_THUMBTRACK\r\n");
			nNewPos = nTrackPos;
			break;
		}

//		TRACE(L"scroll. %d\r\n", nNewPos);
		if (nNewPos > nMax - nPage + 1)
			nPos = nMax - nPage + 1;
		else if (nNewPos >= 0)
			nPos = nNewPos;
		else
			nPos = 0;
	}
	return nPos;
}

void WinUtil::ProcessScrollEvent(CWnd& wnd, int nType, neuro_32 nSBCode, neuro_u32 nTrackPos)
{
	SCROLLINFO scInfo;
	wnd.GetScrollInfo(nType, &scInfo);

	neuro_32 nPos = GetScrollInfo(nSBCode, scInfo.nPage, scInfo.nMax, scInfo.nPos, nTrackPos);
//	TRACE(L"scroll. %d\r\n", nPos);

	wnd.SetScrollPos(nType, nPos);

	wnd.Invalidate();
}

void WinUtil::AdjustListBoxHeight(CWnd& parent, CListBox& listBox)
{
	if (listBox.GetCount() == 0)
		return;

	int client_height = listBox.GetItemHeight(0)*listBox.GetCount();
	CRect rcListBox;
	listBox.GetWindowRect(rcListBox);
	parent.ScreenToClient(rcListBox);

	CRect rcClient;
	listBox.GetClientRect(rcClient);
	rcListBox.bottom = rcListBox.top + rcListBox.Height() - rcClient.Height() + client_height;
	listBox.MoveWindow(rcListBox);
}

void WinUtil::ResizeListBoxHScroll(CListBox& listbox)
{
	CDC* pDC = listbox.GetDC();

	TEXTMETRIC tm;
	pDC->GetTextMetrics(&tm);

	CString str;
	CSize   sz;
	int     dx = 0;
	for (int i = 0; i < listbox.GetCount(); i++)
	{
		listbox.GetText(i, str);
		sz = pDC->GetTextExtent(str);
		sz.cx += +tm.tmAveCharWidth;

		if (sz.cx > dx)
			dx = sz.cx;
	}
	listbox.ReleaseDC(pDC);

	// Set the horizontal extent only if the current extent is not large enough.
	//	if (GetHorizontalExtent() < dx)
	//	{
	listbox.SetHorizontalExtent(dx);
	listbox.Invalidate(FALSE);
}

void WinUtil::ResizeListControlHeader(CListCtrl& listCtrl)
{
	CHeaderCtrl* pHeaderCtrl = listCtrl.GetHeaderCtrl();
	int nColumn = pHeaderCtrl->GetItemCount();
	for (int i = 0; i < nColumn; i++)
		listCtrl.SetColumnWidth(i, LVSCW_AUTOSIZE_USEHEADER);
}

void WinUtil::DeleteAllColumns(CListCtrl& listCtrl)
{
	CHeaderCtrl* pHeaderCtrl = listCtrl.GetHeaderCtrl();
	int nColumn = pHeaderCtrl->GetItemCount();
	for (int i = 0; i < nColumn; i++)
		listCtrl.DeleteColumn(0);
}

/*
void CMyListCtrl::SetColumnWidth(int nCol)
{
	int nWidth = GetStoredWidth( nCol );
	if( nWidth > m_nMinColWidth )
		CListCtrl::SetColumnWidth( nCol, nWidth );
	else
		AutoSizeColumn( nCol );
}

int CMyListCtrl::GetStoredWidth(int nCol)
{
	CString strValue = AfxGetApp()->GetProfileString(
	m_strSection, m_strEntry, m_strDefault );

	CString strSubString;
	AfxExtractSubString( strSubString, strValue, nCol, _T(','));

	return _ttoi( strSubString );
}

void CMyListCtrl::AutoSizeColumn(int nCol)
{
	SetRedraw(false);

	int nMinCol = nCol < 0 ? 0 : nCol;
	int nMaxCol = nCol < 0 ? GetColumnCount() - 1 : nCol;

	for (nCol = nMinCol; nCol <= nMaxCol; nCol++)
	{
		CListCtrl::SetColumnWidth(nCol, LVSCW_AUTOSIZE);
		int wc1 = GetColumnWidth(nCol);

		CListCtrl::SetColumnWidth(nCol, LVSCW_AUTOSIZE_USEHEADER);
		int wc2 = GetColumnWidth(nCol);
		int wc = max(m_nMinColWidth, max(wc1, wc2));

		if (wc > m_nMaxColWidth)
			wc = m_nMaxColWidth;

		// set the column width.
		CListCtrl::SetColumnWidth(nCol, wc);
	}

	SetRedraw();
}
*/
