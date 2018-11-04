#include "stdafx.h"

#include "GraphWnd.h"
#include "GraphicUtil.h"

#include "GraphicUtil.h"
#include "TextDraw.h"

using namespace np::gui;
using namespace np::gui::win32;

CGraphWnd::CGraphWnd(GraphDataSourceAbstract& graphSource)
: m_graphSource(graphSource)
{
	memset(&m_labelFont, 0, sizeof(LOGFONT));
	m_labelFont.lfHeight = 14;
	wcscpy(m_labelFont.lfFaceName, L"SimSun");

	memcpy(&m_curLabelFont, &m_labelFont, sizeof(LOGFONT));
//	m_curLabelFont.lfHeight = 15;
	m_curLabelTextColor = RGB(0, 0, 0);
	m_curLabelBkColor = RGB(209, 209, 209);

	m_bLeftVertLabel = true;

	m_nInnerGap = 10;

	m_nXValueWidth=2;
	m_nXUnitRectWidth=m_nXValueWidth+2;

	m_nXLabelHeight=30;
	m_nYLabelWidth=80;

	m_graphScopeBrush.CreateSolidBrush(RGB(255,255,255));
	m_selectedBarBrush.CreateSolidBrush(RGB(68, 17, 242));

	m_cur_bar_clr = RGB(82, 73, 163);
	m_select_bar_clr = RGB(68, 17, 242);

	m_selected_bar_pos = -1;
}

CGraphWnd::~CGraphWnd()
{
}

BEGIN_MESSAGE_MAP(CGraphWnd, CWnd)
	ON_WM_PAINT()
	ON_WM_ERASEBKGND()
	ON_WM_SIZE()
	ON_WM_HSCROLL()
	ON_WM_MOUSEMOVE()
END_MESSAGE_MAP()

BOOL CGraphWnd::PreCreateWindow(CREATESTRUCT& cs)
{
	cs.style|=WS_HSCROLL;	// 요게 잘 안되면 OnCreate에서 하자

	return CWnd::PreCreateWindow(cs);
}

void CGraphWnd::PreSubclassWindow()
{
	CWnd::PreSubclassWindow();
}

BOOL CGraphWnd::OnEraseBkgnd(CDC* pDC)
{
	// TODO: Add your message handler code here and/or call default

	return FALSE;
}

void CGraphWnd::OnSize(UINT nType, int cx, int cy)
{
	RefreshScrollBars();

	__super::OnSize(nType, cx, cy);
}

void CGraphWnd::RefreshScrollBars()
{
	neuro_u32 total = m_graphSource.GetTotalScrollDataCount();
	if (total <= GetViewDataCount())
	{
		SCROLLINFO scInfo;
		memset(&scInfo, 0, sizeof(scInfo));
		SetScrollInfo(SB_HORZ, &scInfo);
		ShowScrollBar(SB_HORZ, FALSE);
		return;
	}

	SCROLLINFO scInfo;
	memset(&scInfo, 0, sizeof(scInfo));
	GetScrollInfo(SB_HORZ, &scInfo);
	scInfo.cbSize = sizeof(scInfo);
	scInfo.fMask = SIF_ALL;
	scInfo.nMin = 0;
	scInfo.nMax = total - 1;
	scInfo.nPage = GetViewDataCount();
	scInfo.nTrackPos = 1;
	scInfo.nPos = 0;

	SetScrollInfo(SB_HORZ, &scInfo);
	ShowScrollBar(SB_HORZ, TRUE);
}

#include "gui/win32/WinUtil.h"
void CGraphWnd::OnHScroll(UINT nSBCode, UINT nTrackPos, CScrollBar* pScrollBar)
{
	WinUtil::ProcessScrollEvent(*this, SB_HORZ, nSBCode, nTrackPos);

	__super::OnHScroll(nSBCode, nTrackPos, pScrollBar);
}

void CGraphWnd::OnMouseMove(UINT nFlags, CPoint point)
{
	Invalidate();	// 현재 cursor 위치의 정보를 표시하기 위해

	CWnd::OnMouseMove(nFlags, point);
}

void CGraphWnd::EnsureDataPos(neuro_64 pos)
{
	if (pos<GetViewDataStart() || pos>(GetViewDataStart() + GetViewDataCount()))
	{
		if (pos>GetViewDataCount() / 2)
			SetScrollPos(SB_HORZ, pos - GetViewDataCount() / 2);
		else
			SetScrollPos(SB_HORZ, 0);
	}
}

void CGraphWnd::ChangedSource()
{
	SCROLLINFO scInfo;
	GetScrollInfo(SB_HORZ, &scInfo);
	SetScrollPos(SB_HORZ, scInfo.nMin);
	RefreshScrollBars();

	Invalidate();
}

void CGraphWnd::SelectCurDataPos(neuro_u64 pos)
{
	m_selected_bar_pos = pos; 

	if (GetSafeHwnd()!=NULL)
		Invalidate();

	EnsureDataPos(pos);
}

int CGraphWnd::GetViewDataStart() const
{
	SCROLLINFO scInfo;
	memset(&scInfo, 0, sizeof(scInfo));
	if(!const_cast<CGraphWnd*>(this)->GetScrollInfo(SB_HORZ, &scInfo))
		return 0;

	if (scInfo.nPage == 0)
		return 0;
	return scInfo.nPos;
}

int CGraphWnd::GetViewDataCount() const
{
	CRect clientRect;
	GetClientRect(&clientRect);

	CRect graphFullRect;
	GetGraphFullRect(clientRect, graphFullRect);
	//	return NP_Util::CalculateCountPer(graphFullRect.Width(), m_nXUnitRectWidth);
	return graphFullRect.Width() / m_nXUnitRectWidth;
}

neuro_64 CGraphWnd::GetDataPosFromWindowPoint(int start_x, int x) const
{
	return (x - start_x) / m_nXUnitRectWidth + GetViewDataStart();
}

CRect CGraphWnd::GetBarRect(const CRect& rcGraph, int nDataPos) const
{
	if (nDataPos<0 || nDataPos<GetViewDataStart())
		return CRect(0, 0, 0, 0);

	__int32 nBarPos = (nDataPos - GetViewDataStart())*GetXUnitRectWidth();

	CRect rcBar;
	rcBar.left = rcGraph.left + nBarPos;
	rcBar.right = rcBar.left + GetXUnitRectWidth();
	rcBar.top = rcGraph.top;
	rcBar.bottom = rcGraph.bottom;
	return rcBar;
}

void CGraphWnd::DrawRect(CDC& dc, const CRect& rc)
{
	dc.Rectangle(rc);
	/*
	dc.MoveTo(rc.left, rc.top);
	dc.LineTo(rc.right, rc.top);
	dc.LineTo(rc.right, rc.bottom);
	dc.LineTo(rc.left, rc.bottom);
	dc.LineTo(rc.left, rc.top);
	//*/
}

void CGraphWnd::OnPaint()
{
	CPaintDC paintDC(this); // device context for painting

	CRect clientRect;
	GetClientRect(&clientRect);

	if (clientRect.right <= m_nYLabelWidth)
		return;

	if(clientRect.left>clientRect.right || clientRect.top>clientRect.bottom)
		return;

	const int nAvailableViewDataCount=GetViewDataCount();
	if(nAvailableViewDataCount<0)
		return;

	CMemDC memDC(paintDC, clientRect);

	CDC& dc=memDC.GetDC();
//	dc.SetBkMode(TRANSPARENT);
	dc.SetBkMode(OPAQUE);

	CBrush* pOldbrush = dc.SelectObject(&m_graphScopeBrush);

	DrawRect(dc, clientRect);

//    int log_pix = GetDeviceCaps(dc.m_hAttribDC, LOGPIXELSY);
//	m_labelFont.lfHeight = 10*log_pix;

	CFont font;
	font.CreateFontIndirect(&m_labelFont);
	CFont* pOldFont=dc.SelectObject(&font);

	neuro_64 nStart = GetViewDataStart();

	CRect graphFullRect;
	GetGraphFullRect(clientRect, graphFullRect);
	DrawRect(dc, graphFullRect);

	if (m_graphSource.IsValid(nStart, nAvailableViewDataCount))
	{
		CRect graphDrawRect = graphFullRect;
//		graphDrawRect.DeflateRect(m_nInnerGap, 0);

		neuro_64 cur_data_pos = -1;
		{
			CPoint pt;
			if (GetCursorPos(&pt))
			{
				ScreenToClient(&pt);
				cur_data_pos = GetDataPosFromWindowPoint(graphDrawRect.left, pt.x);
			}
		}

		neuro_u32 max_ylabel = graphDrawRect.Height() / TextDraw::CalculateTextSize(dc, NP_RECT(0, 0, 100, 100), L"0").height / 10;

		_graph_frame graphFrame;
		m_graphSource.GetViewData(nStart, nAvailableViewDataCount, max_ylabel, cur_data_pos, graphFrame);

		DrawGraph(dc, clientRect, graphDrawRect, graphFrame);

		CRect rcHorzLabel = graphDrawRect;
		rcHorzLabel.top = clientRect.bottom - m_nXLabelHeight;
		rcHorzLabel.bottom = clientRect.bottom;

		DrawXLabel(dc, rcHorzLabel, graphFrame.xLabelVector);

		if (graphFrame.has_cur_pos)
		{
			CFont font;
			font.CreateFontIndirect(&m_curLabelFont);
			CFont* pOldFont = dc.SelectObject(&font);
			COLORREF prev_color = dc.SetTextColor(m_curLabelTextColor);
			COLORREF prev_bk_color = dc.SetBkColor(m_curLabelBkColor);

			DrawXLabel(dc, rcHorzLabel, { graphFrame.curpos_xLabel }, true);

			dc.SetTextColor(prev_color);
			dc.SetBkColor(prev_bk_color);
			dc.SelectObject(pOldFont);
		}

		if (m_selected_bar_pos > 0)
		{
			CRect rcBar = GetBarRect(graphDrawRect, m_selected_bar_pos);
			if (rcBar.right <= graphDrawRect.right)
			{
				dc.SelectObject(&m_selectedBarBrush);
				dc.Rectangle(rcBar);
			}
		}
	}

	dc.SelectObject(pOldbrush);
	dc.SelectObject(pOldFont);
}

void CGraphWnd::DrawGraph(CDC& dc, const CRect& clientRect, const CRect& graphDrawRect, const _graph_frame& graphFrame)
{
	const std::vector<_graph_view>& graphViewVector = graphFrame.graphViewVector;
	neuro_64 nView = graphViewVector.size();
	if(nView==0)
		return;

	// 값이 별로 없으면 좁게 그리자
	neuro_float deflate_ratio = 0.f;
	if (m_graphSource.GetTotalScrollDataCount() < 5)
		deflate_ratio = 0.2;
	else if (m_graphSource.GetTotalScrollDataCount() < 10)
		deflate_ratio = 0.1;

	HDC hdc = dc.GetSafeHdc();

	int nLastGraphBottom = graphDrawRect.top;
	for (neuro_64 iView = 0; iView<nView; iView++)
	{
		const _graph_view& graphView=graphViewVector[iView];

		const int nGraphHeight = graphDrawRect.Height()*graphView.heightRatio;
//		const int nGraphHeight=graphDrawRect.Height()/nGraph;

		CRect rcGraph;
		rcGraph.left = graphDrawRect.left;
		rcGraph.right = graphDrawRect.right;
		rcGraph.top=nLastGraphBottom;
		rcGraph.bottom=rcGraph.top+nGraphHeight;

		dc.SelectObject(&m_graphScopeBrush);

		nLastGraphBottom=rcGraph.bottom;

		GraphicUtil::DrawLine(hdc, { rcGraph.left, rcGraph.top }, { rcGraph.right, rcGraph.top }, RGB(0, 0, 0), 1);
		GraphicUtil::DrawLine(hdc, { rcGraph.left, rcGraph.bottom }, { rcGraph.right, rcGraph.bottom }, RGB(0, 0, 0), 1);

		rcGraph.DeflateRect(0, m_nInnerGap);
		rcGraph.DeflateRect(0, rcGraph.Height()*deflate_ratio);

		neuro_float nValueUnit=0;
		{
			neuron_value nValueDivide=graphView.upper_boundary-graphView.lower_boundary;
			if(nValueDivide>0)
				nValueUnit = ((neuro_float)rcGraph.Height() - (neuro_float)m_nXUnitRectWidth) / (neuro_float)nValueDivide;
			else
				nValueUnit=0;
		}

		for(neuro_u32 iLine=0;iLine<graphView.graphLineVector.size();iLine++)
		{
			const _graph_line& valueGroup=graphView.graphLineVector[iLine];
			if (valueGroup.valueArray.size()==0)
				continue;

			Gdiplus::Pen line_pen(GraphicUtil::Transform(valueGroup.clr));
			Gdiplus::Pen pt_pen(GraphicUtil::Transform(RGB(0,0,0)), 2);

			long cur_left = rcGraph.left + (m_nXUnitRectWidth - m_nXValueWidth) / 2;
			NP_POINT movingLineStart;
			for(size_t iData=0;iData<valueGroup.valueArray.size();iData++)
			{
				neuron_value value = valueGroup.valueArray[iData] - graphView.lower_boundary;

				CRect valueRect;
				valueRect.left = cur_left;
				valueRect.right=valueRect.left+m_nXValueWidth;
				valueRect.top=GetValuePosY(rcGraph, nValueUnit, value);

				if (graphView.shapeType == _graph_view::_shape_type::bar)
				{
					valueRect.bottom = rcGraph.bottom;
					dc.Rectangle(valueRect);
				}
				else if (graphView.shapeType == _graph_view::_shape_type::line)
				{
					NP_POINT movingLineEnd(valueRect.left+(valueRect.Width())/2, valueRect.top);
					if (iData > 0)
						GraphicUtil::DrawLine(hdc, movingLineStart, movingLineEnd, line_pen);

					GraphicUtil::DrawPoint(hdc, movingLineEnd, pt_pen);

					movingLineStart=movingLineEnd;
				}
				else
				{
					valueRect.top -= m_nXUnitRectWidth / 2;
					valueRect.bottom = valueRect.top + m_nXUnitRectWidth;
					dc.Ellipse(valueRect);
				}

				cur_left += m_nXUnitRectWidth;
			}
		}

		// y label을 그리자
		DrawYLabel(dc, clientRect, rcGraph, nValueUnit, graphView.lower_boundary, graphView.yLabelArray);

		CFont font;
		font.CreateFontIndirect(&m_curLabelFont);
		CFont* pOldFont = dc.SelectObject(&font);
		COLORREF prev_color = dc.SetTextColor(m_curLabelTextColor);
		COLORREF prev_bk_color = dc.SetBkColor(m_curLabelBkColor);

//		if(graphView.graphLineVector.size()>0 && graphView.graphLineVector[0].valueArray.size()>1)
		if(graphFrame.has_cur_pos)
			DrawYLabel(dc, clientRect, rcGraph, nValueUnit, graphView.lower_boundary, graphView.curpos_yLabel_vector, &graphFrame.curpos_xLabel.value);

		dc.SetTextColor(prev_color);
		dc.SetBkColor(prev_bk_color);
		dc.SelectObject(pOldFont);
	}
}

void CGraphWnd::DrawXLabel(CDC& dc, const CRect& rcHorzLabel, const std::vector<_graphLabel>& xLabelVector, bool draw_vert_line)
{
	dc.SelectObject(&m_graphScopeBrush);

	HDC hdc = dc.GetSafeHdc();
	for (INT_PTR iLabel = 0; iLabel<xLabelVector.size(); iLabel++)
	{
		int x = rcHorzLabel.left + xLabelVector[iLabel].value * m_nXUnitRectWidth + m_nXUnitRectWidth / 2;
		if (draw_vert_line)
			GraphicUtil::DrawLine(hdc, { x, 0 }, { x, rcHorzLabel.top }, m_cur_bar_clr, 1);

		NP_2DSHAPE rc;
		rc.sz = TextDraw::CalculateTextSize(dc, NP_RECT(0, 0, 100, 100), xLabelVector[iLabel].label);
		rc.pt.x = x  - rc.sz.width / 2;
		rc.pt.y = rcHorzLabel.top + (rcHorzLabel.Height() - rc.sz.height) / 2;

		GraphicUtil::DrawLine(hdc, { x, rcHorzLabel.top }, { x, rc.pt.y - 2 }, RGB(0,0,0), 1);

		gui::win32::TextDraw::SingleText(dc, rc, xLabelVector[iLabel].label, gui::win32::horz_align::center);
	}
}
void CGraphWnd::DrawYLabel(CDC& dc, const CRect& clientRect, const CRect& rcGraph, neuron_value nValueUnit, neuron_value lower_boundary, const std::vector<_graphLabel>& y_label_vector, const neuron_value* cur_pos_x)
{
	HDC hdc = dc.GetSafeHdc();

	std::vector<_graphLabel>::const_iterator it_ylabel = y_label_vector.begin();
	for (; it_ylabel != y_label_vector.end(); it_ylabel++)
	{
		const _graphLabel& graphLabel = *it_ylabel;
		neuro_float labelValue = graphLabel.value - lower_boundary;

		int y = GetValuePosY(rcGraph, nValueUnit, labelValue);

		NP_SIZE sz = TextDraw::CalculateTextSize(dc, NP_RECT(0, 0, 100, 100), graphLabel.label);

		NP_RECT rc;
		if (cur_pos_x)
		{
			int x = rcGraph.left + *cur_pos_x * m_nXUnitRectWidth + m_nXUnitRectWidth / 2;

			if (x + 5 + sz.width < rcGraph.right)
			{
				rc.left = x + 5;
				rc.right = rc.left + sz.width;
			}
			else
			{
				rc.right = x - 5;
				rc.left = rc.right - sz.width;
			}
			/*
			if (y - sz.height >= rcGraph.top + 1)
			{
				rc.bottom = y - 11;
				rc.top = rc.bottom - sz.height;
			}
			else
			{
				rc.top = y + 11;
				rc.bottom = rc.top + sz.height;
			}
			*/
			rc.top = y - sz.height / 2;
			rc.bottom = rc.top + sz.height;

			Gdiplus::Pen pen(Gdiplus::Color(0, 0, 0), 1.1f);
			gui::win32::GraphicUtil::CompositeLinePen(pen, gui::_line_arrow_type::none, gui::_line_dash_type::dot);
//			GraphicUtil::DrawLine(hdc, line_from, line_to, pen);
		}
		else
		{
			rc.top = y - (sz.height + 2) / 2;
			rc.bottom = rc.top + (sz.height + 2);
			if (m_bLeftVertLabel)
			{
				GraphicUtil::DrawLine(hdc, { rcGraph.left - 3, y }, { rcGraph.left, y }, RGB(0,0,0), 1);

				rc.left = clientRect.left;
				rc.right = rcGraph.left - 5;
			}
			else
			{
				GraphicUtil::DrawLine(hdc, { rcGraph.right, y }, { rcGraph.right + 3, y }, RGB(0, 0, 0), 1);

				rc.left = rcGraph.right + 5;
				rc.right = clientRect.right;
			}
		}
		gui::win32::TextDraw::SingleText(dc, rc, graphLabel.label, gui::win32::horz_align::right, true, true);
	}
}

int CGraphWnd::GetValuePosY(const CRect& rcGraph, neuron_value nValueUnit, neuron_value value) const
{
	return rcGraph.bottom-nValueUnit*value;
}

void CGraphWnd::GetGraphFullRect(const CRect& clientRect, CRect& rc) const
{
	if (m_bLeftVertLabel)
	{
		rc.left = clientRect.left + m_nYLabelWidth;
		rc.right = clientRect.right;
	}
	else
	{
		rc.left = clientRect.left;
		rc.right = clientRect.right - m_nYLabelWidth;
	}
	rc.top=clientRect.top;
	rc.bottom=clientRect.bottom-m_nXLabelHeight;
}
