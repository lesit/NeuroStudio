#include "stdafx.h"

#include "AbstractNNWnd.h"

#include "util/StringUtil.h"

#include "gui/Win32/WinUtil.h"
#include "gui/Win32/TextDraw.h"

#include "project/NeuroStudioProject.h"
#include "desc/LayerDesc.h"

#include "math.h"

using namespace np::gui;
using namespace np::studio;

AbstractNNWnd::AbstractNNWnd(const network::NetworkMatrix& network_matrix, const std::vector<_CLIPBOARDFORMAT_INFO>& cf_vector)
: CMappingWnd(cf_vector), m_network_matrix(network_matrix)
{
	m_mouseoverLayer = NULL;
	m_mouseoverLink = NULL;

	m_bMouseLButtonDown = false;

	ClearSelect();
}

AbstractNNWnd::~AbstractNNWnd()
{
}

NP_SIZE AbstractNNWnd::GetScrollTotalViewSize() const
{
	return NP_SIZE(m_network_matrix.GetMatrixWidth(), m_network_matrix.GetMatrixHeight());
}

neuro_u32 AbstractNNWnd::GetScrollMoving(bool is_horz) const
{
	return is_horz ? m_network_matrix.GetGridLayout().grid_size.width / 5 : m_network_matrix.GetGridLayout().grid_size.height / 5;
}

void AbstractNNWnd::Draw(CDC& dc, CRect rcClient)
{
	HDC hdc = dc.GetSafeHdc();
/*	
	{
		// 좌측 gradient
		TRIVERTEX		vert[2];
		GRADIENT_RECT	gRect;
		vert[0].x = 0;
		vert[0].y = 0;
		vert[0].Red = 0xff00;
		vert[0].Green = 0xff00;
		vert[0].Blue = 0xff00;
		vert[0].Alpha = 0x0000;

		vert[1].x = rcClient.Width();
		vert[1].y = rcClient.Height();
		vert[1].Red = 0x1f00;	//  31
		vert[1].Green = 0x9500;	// 149
		vert[1].Blue = 0xc500;	// 197
		vert[1].Alpha = 0x0000;

		gRect.UpperLeft = 0;
		gRect.LowerRight = 1;
		GradientFill(hdc, vert, 2, &gRect, 1, GRADIENT_FILL_RECT_H);
	}
//*/

	dc.SetBkMode(TRANSPARENT);

	// draw links
	_link_map::const_iterator it_link = m_network_matrix.GetLinkMap().begin();
	_link_map::const_iterator end_link = m_network_matrix.GetLinkMap().end();
	for (; it_link != end_link; it_link++)
	{
		const _LINK_INFO& link = it_link->second;

		NP_POINT pt_from = ViewportToWnd(link.line.bezier.start);
		NP_POINT pt_to = ViewportToWnd(link.line.bezier.points[link.line.bezier.points.size()-1].pt);

		// 연결선이 화면에 조금이라도 보이는 것만 그린다.
		if (pt_from.x > rcClient.right || pt_to.x < rcClient.left)
			continue;

		if(pt_from.y < rcClient.top && pt_to.y < rcClient.top ||
			pt_from.y > rcClient.bottom && pt_to.y > rcClient.bottom)
			continue;

		_line_type line_type;
		if (m_selected_unit.link == &link)
			line_type = _line_type::select;
		else if (m_mouseoverLink == &link)
			line_type = _line_type::mouseover;
		else
			line_type = _line_type::normal;

		DrawMappingLine(hdc, line_type, link.line);
	}

	CBrush* pOldBrush = dc.GetCurrentBrush();
	COLORREF prev_textcolor = dc.GetTextColor();
	COLORREF prev_bkcolor = dc.GetBkColor();

	const gui::_GRID_LAYOUT& grid_layout = m_network_matrix.GetGridLayout();

	// draw layers
	NP_RECT vp_rc = WndToViewport(NP_RECT(rcClient.left, rcClient.top, rcClient.right, rcClient.bottom));
	MATRIX_SCOPE matrix_scope = m_network_matrix.GetMatrixScope(vp_rc, true);
	neuro_u32 last_level = min((neuro_u32)m_network_matrix.GetLevelCount(), matrix_scope.second.level);
	for (neuro_u32 level= matrix_scope.first.level; level < last_level; level++)
	{
		std::wstring levelLabel = util::StringUtil::Transform<wchar_t>(level);
		NP_RECT rcLevelLabel = m_network_matrix.GetGridRect({ level, 0 });
		rcLevelLabel.bottom = rcLevelLabel.top;
		rcLevelLabel.top = 5;
		rcLevelLabel = ViewportToWnd(rcLevelLabel);
		rcLevelLabel.left += grid_layout.item_margin.width + 20;

		dc.SetTextColor(RGB(164,177,193));
//		dc.SetBkColor(RGB(255, 255, 255));
		gui::win32::TextDraw::SingleText(dc, rcLevelLabel, levelLabel, gui::win32::horz_align::left);

		neuro_u32 last_row = min(m_network_matrix.GetRowCount(level), matrix_scope.second.row);
		for (neuro_u32 row = matrix_scope.first.row; row < last_row; row++)
		{
			AbstractLayer* layer = m_network_matrix.GetLayer(level, row);
			if (layer == NULL)
				continue;

			MATRIX_POINT layer_mp(level, row);

			COLORREF layer_txt_clr = RGB(109, 71, 199); 
			COLORREF desc_txt_clr = RGB(90, 90, 90);

			CBrush* brush;
			if (m_selected_unit.layer == layer || m_selected_scope.InScope(level, row))
			{
				brush = &m_select_brush;
				layer_txt_clr = RGB(255, 255, 255);
				desc_txt_clr = RGB(241, 239, 245);
			}
			else if (m_mouseoverLayer == layer)
				brush = &m_cur_layer_brush;
			else
				brush = &m_normal_layer_brush;

			dc.SelectObject(brush);

			LOGBRUSH lb;
			brush->GetLogBrush(&lb);
//			dc.SetBkColor(lb.lbColor);
			dc.SetTextColor(layer_txt_clr);

			NP_RECT rcLabel = ViewportToWnd(m_network_matrix.GetLayerRect(layer_mp));
			dc.RoundRect(rcLabel.left, rcLabel.top, rcLabel.right, rcLabel.bottom, 10, 10);

			rcLabel.DeflateRect(4, 4);
			std::wstring sub_desc = str_rc::LayerDesc::GetDesignSubDesc(*layer);

			std::wstring desc = str_rc::LayerDesc::GetDesignDesc(*layer);
			if (sub_desc.empty())
			{
				gui::win32::TextDraw::SingleText(dc, rcLabel, desc, gui::win32::horz_align::center);
			}
			else
			{
				NP_SIZE sz = gui::win32::TextDraw::CalculateTextSize(dc, rcLabel, desc);
				gui::win32::TextDraw::SingleText(dc, rcLabel, desc, gui::win32::horz_align::center, false);

				dc.SetTextColor(desc_txt_clr);

				rcLabel.top += sz.height+2;
				gui::win32::TextDraw::MultiText(dc, rcLabel, sub_desc, gui::win32::horz_align::center);
			}
		}
	}

	dc.SetTextColor(prev_textcolor);
	dc.SetBkColor(prev_bkcolor);
	dc.SelectObject(pOldBrush);
}

const _LINK_INFO* AbstractNNWnd::LinkHitTest(const NP_POINT& point) const
{
	NP_POINT vp_pt = WndToViewport(point);

	_link_map::const_iterator it = m_network_matrix.GetLinkMap().begin();
	_link_map::const_iterator end = m_network_matrix.GetLinkMap().end();
	for (; it != end; it++)
	{
		const _LINK_INFO& link = it->second;
		if(LineHitTest(vp_pt, link.line))
			return &link;
	}

	return false;
}

void AbstractNNWnd::ClearSelect()
{
	m_selected_unit.Initialize();
	m_selected_scope.Initialize();
}

void AbstractNNWnd::SelectLayer(AbstractLayer* layer)
{
	ClearSelect();

	m_selected_unit.layer = layer;
	if(layer)
		EnsureVisible(*layer);

	RefreshDisplay();
}

void AbstractNNWnd::EnsureVisible(const AbstractLayer& layer)
{
	MATRIX_POINT mp = m_network_matrix.GetLayerMatrixPoint(layer);
	NP_RECT layer_rc = m_network_matrix.GetLayerRect(mp);

	CRect rcClient;
	GetClientRect(rcClient);
	NP_RECT vp_rc = WndToViewport(NP_RECT(rcClient.left, rcClient.top, rcClient.right, rcClient.bottom));
	MATRIX_SCOPE matrix_scope = m_network_matrix.GetMatrixScope(vp_rc, true);

	if (mp.level >= matrix_scope.first.level && mp.level < matrix_scope.second.level
		&& mp.row >= matrix_scope.first.row && mp.row < matrix_scope.second.row)
		return;

	NP_POINT vp = GetViewport();
	if (mp.level < matrix_scope.first.level || mp.level >= matrix_scope.second.level)
		vp.x = layer_rc.left;

	if (mp.row < matrix_scope.first.row || mp.row >= matrix_scope.second.row)
		vp.y = layer_rc.top;

	SetViewport(vp);
}

_POS_INFO_IN_LAYER AbstractNNWnd::SelectNeuroUnit(NP_POINT point)
{
	TRACE(L"SelectNeuroUnit. %d, %d\r\n", point.x, point.y);

	ClearSelect();

	_POS_INFO_IN_LAYER pos_info;
	pos_info.Initialize();

	const _LINK_INFO* link = LinkHitTest(point);
	if (link != NULL)
	{
		m_selected_unit.link = link;
	}
	else if (m_network_matrix.LayerHitTest(WndToViewport(point), pos_info)
		&& pos_info.pos_in_grid == _POS_INFO_IN_LAYER::_pos_in_grid::layer
		&& pos_info.layer != NULL)
		m_selected_unit.layer = pos_info.layer;

	AfterNetworkSelected(point);

	RefreshDisplay();	// 새로 선택된 link 또는 layer를 다시 그리기 위해
	return pos_info;
}


void AbstractNNWnd::SelectMultiLayers(const NP_2DSHAPE& rc)
{
	ClearSelect();

	m_selected_scope = m_network_matrix.GetMatrixScope(WndToViewport(rc), false);
#ifdef _DEBUG
	if (rc.sz.width == 0 || rc.sz.height == 0)
	{
		m_selected_scope = m_network_matrix.GetMatrixScope(WndToViewport(rc), false);
	}
#endif

	for (neuro_u32 level = m_selected_scope.first.level; level < m_selected_scope.second.level; level++)
	{
		for (neuro_u32 row = m_selected_scope.first.row; row < m_selected_scope.second.row; row++)
		{
			if (m_network_matrix.GetLayer(level, row) != NULL)
			{
				RefreshDisplay();
				return;
			}
		}
	}
	m_selected_scope.Initialize();
}

void AbstractNNWnd::MouseLClickEvent(bool bMouseDown, NP_POINT point)
{
	TRACE(_T("MouseLClickEvent : %s\r\n"), bMouseDown ? L"down" : L"up");

	SelectNeuroUnit(point);

	if (bMouseDown)
	{
		/*	직전에 context menu를 출력했고 메뉴 클릭없이 다른곳에 마우스 클릭하면
			WM_LBUTTONDOWN 만 오고 버튼을 놓았을 때 WM_LBUTTONUP가 아닌 WM_MOUSEMOVE 가 온다.
			그렇다고, context menu가 출력됐는지 확인해서 그후에 WM_LBUTTONDOWN이 왔을때 무시하면 
			MouseMoveEvent 에서 m_bMouseLButtonDown=false 상태가 되기 때문에 layer 연결이나 멀티 선택을 할수 없다.
			어짜피 마우스 움직였을때도 처리해야 하니까 그냥 냅둔다.
		*/
		CWnd* focused = GetFocus();
		if (focused == this || focused == GetParent())
			m_bMouseLButtonDown = true;
		else
			TRACE(L"MouseLClickEvent. not focused.");
	}
	else if (m_bMouseLButtonDown)	// 이 화면에서 마우스를 누르고 나서 떼었을때.
	{
		m_bMouseLButtonDown = false;

	}
	OnLClickedUnit(bMouseDown, point);
}

void AbstractNNWnd::MouseRClickEvent(bool bMouseDown, NP_POINT point)
{
	TRACE(_T("MouseRClickEventk : %s\r\n"), bMouseDown ? L"down" : L"up");

	m_bMouseLButtonDown = false;

	SelectNeuroUnit(point);

	OnRClickedUnit(bMouseDown, point);
}

void AbstractNNWnd::ContextMenuEvent(NP_POINT point)
{
	TRACE(L"ContextMenuEvent\r\n");

	_POS_INFO_IN_LAYER pos_info = SelectNeuroUnit(point);

	OnContextMenu(point, pos_info);

	TRACE(L"\r\nContextMenuEvent. end\r\n");
}

void AbstractNNWnd::MouseMoveEvent(NP_POINT point)
{
//	TRACE(L"Mouse move\r\n");

	const AbstractLayer* prev_layer = m_mouseoverLayer;;

	_POS_INFO_IN_LAYER cur_info;
	m_network_matrix.LayerHitTest(WndToViewport(point), cur_info);

	m_mouseoverLayer = NULL;
	if(cur_info.pos_in_grid == _POS_INFO_IN_LAYER::_pos_in_grid::layer && cur_info.layer != NULL)
		m_mouseoverLayer = cur_info.layer;

	const _LINK_INFO* prev_Link = m_mouseoverLink;
	m_mouseoverLink = LinkHitTest(point);

	AfterNetworkMouseMove(point);

	if (m_bMouseLButtonDown)
	{
		if (m_selected_unit.layer!=NULL)	// layer에서 drag를 시작했을 경우
		{
			MATRIX_POINT layer_mp = m_network_matrix.GetLayerMatrixPoint(*m_selected_unit.layer);
			BeginDragLayer(point, layer_mp);
		}
		else	// left button 누른 상태로 그냥 drag 했을 경우
		{
			NP_2DSHAPE rc;
			RectTracker(point, rc);

			SelectMultiLayers(rc);
		}
		// 모든 dragdrop은 위의 drag.DragDrop 과 RectTracker 에서 시작되고 끝나기 때문에 아래 플래그를 false로 설정해줘야 한다.
		m_bMouseLButtonDown = false;
	}
	else if (prev_layer != m_mouseoverLayer)
	{
		//		TRACE(_T("mose move : over unit changed\r\n"));
	}
	else if (prev_Link != m_mouseoverLink)
	{
		//		TRACE(_T("mose move : over link changed\r\n"));
	}
	else
		return;

	RefreshDisplay();
}
