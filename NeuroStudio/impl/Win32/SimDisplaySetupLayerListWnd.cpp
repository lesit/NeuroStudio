#include "stdafx.h"

#include "SimDisplaySetupLayerListWnd.h"
#include "SimDisplaySetupWnd.h"

#include "NeuroKernel/network/AbstractLayer.h"
#include "gui/Win32/TextDraw.h"

SimDisplaySetupLayerListWnd::SimDisplaySetupLayerListWnd(project::NeuroStudioProject& project
	, const network::NetworkMatrix& network_matrix, AbstractBindingViewManager& binding_view)
	: DataViewManager(binding_view)
	, m_network(*project.GetNSManager().GetNetwork())
	, m_network_matrix(network_matrix)
	, m_layer_display_info_map(project.GetSimManager().GetLayerDisplayInfoMap())
{
	m_grid_layout.SetLayout({ 10, 10, 10, 10 }, { 70, 30 }, { 15, 15 });

	m_bMouseLButtonDown = false;
	m_mouseoverLayout = NULL;
	ResetSelect();
}

void SimDisplaySetupLayerListWnd::LoadView()
{
	std::vector<neuro_u32> remove_display_vector;

	_layer_display_info_map::const_iterator it = m_layer_display_info_map.begin();
	for (; it != m_layer_display_info_map.end(); it++)
	{
		network::AbstractLayer* layer = m_network.FindLayer(it->first);
		if (layer == NULL)
		{
			remove_display_vector.push_back(it->first);
			continue;
		}

		_LAYOUT_INFO inserted;
		InsertMatrixDisplayInfo(*layer, it->second, inserted);
	}
	for (neuro_u32 i = 0; i < remove_display_vector.size(); i++)
		m_layer_display_info_map.erase(remove_display_vector[i]);

	ResetSelect();
	if (m_matrix_display_vector.size() > 0)
	{
		m_selectedLayout.row = m_selectedLayout.col;
		m_selectedLayout.layout = &m_matrix_display_vector[0][0];
	}
	RefreshScrollBars();

	ShowConfigProperty();
}

void SimDisplaySetupLayerListWnd::SaveView()
{
	m_layer_display_info_map.clear();
	for (neuro_u32 row = 0, n = m_matrix_display_vector.size(); row < n; row++)
	{
		const _layer_display_setup_row_vector& col_vector = m_matrix_display_vector[row];
		for (neuro_u32 col = 0; col < col_vector.size(); col++)
		{
			const LayerDisplaySetup& layout = col_vector[col];
			m_layer_display_info_map[layout.layer->uid] = layout.display;
		}
	}
}

bool SimDisplaySetupLayerListWnd::InsertMatrixDisplayInfo(const network::AbstractLayer& layer, const _LAYER_DISPLAY_INFO& display_info, _LAYOUT_INFO& inserted)
{
	inserted.layout = NULL;

	LayerDisplaySetup info;
	info.layer = &layer;
	info.display = display_info;
	info.mp = m_network_matrix.GetLayerMatrixPoint(layer);
	if (m_matrix_display_vector.size() == 0)
	{
		m_matrix_display_vector.push_back({ info });

		inserted.row = inserted.col = 0;
		inserted.layout = &m_matrix_display_vector[0][0];

		return true;
	}

	inserted.row = FindLayoutRow(info.mp, layer);
	inserted.col = 0;

	_layer_display_setup_row_vector& col_vector = m_matrix_display_vector[inserted.row];
	if (info.mp.level == col_vector[0].mp.level)
	{
		for (neuro_u32 col = 0; col < col_vector.size(); col++)
		{
			if (info.mp.row > col_vector[col].mp.row)
			{
				col_vector.insert(col_vector.begin() + col + 1, info);

				inserted.col = col + 1;
				break;
			}
		}
	}
	else
	{
		if (info.mp.level > col_vector[0].mp.level)
			++inserted.row;

		m_matrix_display_vector.insert(m_matrix_display_vector.begin() + inserted.row, { info });
	}
	inserted.layout = &m_matrix_display_vector[inserted.row][inserted.col];

	return true;
}

void SimDisplaySetupLayerListWnd::DeleteMatrixDisplayInfo(neuro_u32 row, neuro_u32 col)
{
	if (row >= m_matrix_display_vector.size())
		return;

	_layer_display_setup_row_vector& col_vector = m_matrix_display_vector[row];
	if (col >= col_vector.size())
		return;

	col_vector.erase(col_vector.begin() + col);
	if (col_vector.size() == 0)
		m_matrix_display_vector.erase(m_matrix_display_vector.begin() + row);
}

neuro_u32 SimDisplaySetupLayerListWnd::FindLayoutRow(const MATRIX_POINT& mp, const AbstractLayer& layer) const
{
	neuro_u32 start = 0;
	neuro_u32 last = m_matrix_display_vector.size() - 1;

	while (start<last)
	{
		neuro_u32 middel = start + (last - start) / 2;
		const MATRIX_POINT& middle_mp = m_matrix_display_vector[middel][0].mp;

		if (mp.level == middle_mp.level)
		{
			return middel;
			break;
		}
		else if (mp.level < middle_mp.level)
			last = middel;
		else
			start = middel + 1;
	}
	return start;
}

void SimDisplaySetupLayerListWnd::ToggleDisplayType(const MATRIX_POINT& mp, const AbstractLayer& layer)
{
	_layer_display_info_map::const_iterator it = m_layer_display_info_map.find(layer.uid);
	if (it == m_layer_display_info_map.end())
	{
		_LAYER_DISPLAY_INFO display_info;
		display_info.type = project::_layer_display_type::image;
		display_info.is_onehot_analysis_result = layer.GetLayerType() == network::_layer_type::output;

		_LAYOUT_INFO inserted;
		if (InsertMatrixDisplayInfo(layer, display_info, inserted))
		{
			m_layer_display_info_map[layer.uid] = display_info;

			m_selectedLayout = inserted;
		}
	}
	else
	{
		ResetSelect();
		m_layer_display_info_map.erase(layer.uid);

		neuro_u32 row = FindLayoutRow(mp, layer);
		if (row >= m_matrix_display_vector.size())
			return;

		_layer_display_setup_row_vector& col_vector = m_matrix_display_vector[row];
		if (mp.level == col_vector[0].mp.level)
		{
			for (neuro_u32 col = 0; col < col_vector.size(); col++)
			{
				if (mp.row == col_vector[col].mp.row)
				{
					DeleteMatrixDisplayInfo(row, col);
					ResetSelect();
					break;
				}
			}
		}
	}
	RefreshScrollBars();
	m_binding_view.MakeBindingLineVector();

	ShowConfigProperty();
}

void SimDisplaySetupLayerListWnd::DeleteSelectedDisplay()
{
	m_binding_view.InitSelection();

	if (m_selectedLayout.layout)
		m_layer_display_info_map.erase(m_selectedLayout.layout->layer->uid);
	DeleteMatrixDisplayInfo(m_selectedLayout.row, m_selectedLayout.col);
	RefreshScrollBars();
	m_binding_view.MakeBindingLineVector();
}

void SimDisplaySetupLayerListWnd::ClearAllDisplay()
{
	m_binding_view.InitSelection();

	m_layer_display_info_map.clear();
	m_matrix_display_vector.clear();

	RefreshScrollBars();
	m_binding_view.MakeBindingLineVector();
}

NP_SIZE SimDisplaySetupLayerListWnd::GetScrollTotalViewSize() const
{
	NP_SIZE ret;
	ret.width = m_matrix_display_vector.size() * m_grid_layout.grid_size.width;

	neuro_u32 max_column = 0;
	for (neuro_u32 i = 0, n = m_matrix_display_vector.size(); i < n; i++)
		max_column = max(max_column, m_matrix_display_vector[i].size());

	ret.height = max_column * m_grid_layout.grid_size.height;
	return ret;
}

neuro_u32 SimDisplaySetupLayerListWnd::GetScrollMoving(bool is_horz) const
{
	return is_horz ? m_grid_layout.grid_size.width / 5 : m_grid_layout.grid_size.height / 5;
}

void SimDisplaySetupLayerListWnd::GetBindedModelVector(_binding_source_vector& model_vector) const
{
	CWnd* parent = GetParent();

	NP_POINT pt;
	pt.x = m_grid_layout.item_margin.width;
	for (neuro_u32 row = 0, n = m_matrix_display_vector.size(); row < n; row++)
	{
		pt.y = m_grid_layout.item_margin.height;

		const _layer_display_setup_row_vector& col_vector = m_matrix_display_vector[row];
		for (neuro_u32 col = 0; col < col_vector.size(); col++)
		{
			pt = ViewportToWnd(pt);

			CPoint wnd_pt(pt.x + m_grid_layout.item_size.width / 2, pt.y+ m_grid_layout.item_size.height);
			ClientToScreen(&wnd_pt);
			parent->ScreenToClient(&wnd_pt);	// 부모좌표로 맞추자

			model_vector.resize(model_vector.size() + 1);
			_BINDING_SOURCE_MODEL& binding = model_vector.back();
			binding.from_point = { wnd_pt.x, wnd_pt.y };
			binding.from = const_cast<LayerDisplaySetup*>(&col_vector[col]);
			binding.to = const_cast<network::AbstractLayer*>(col_vector[col].layer);

			pt.y += m_grid_layout.grid_size.height;
		}
		pt.x += m_grid_layout.grid_size.width;
	}
}

void SimDisplaySetupLayerListWnd::OnScrollChanged()
{
	m_binding_view.MakeBindingLineVector();
}

void SimDisplaySetupLayerListWnd::Draw(CDC& dc, CRect rcClient)
{
	HDC hdc = dc.GetSafeHdc();

	CBrush* pOldBrush = dc.GetCurrentBrush();
	COLORREF prev_textcolor = dc.GetTextColor();
	COLORREF prev_bkcolor = dc.GetBkColor();

	NP_2DSHAPE rc;
	rc.sz = m_grid_layout.item_size;

	rc.pt.x = m_grid_layout.item_margin.width;
	for (neuro_u32 row =0; row<m_matrix_display_vector.size(); row++)
	{
		rc.pt.y = m_grid_layout.item_margin.height;

		_layer_display_setup_row_vector& col_vector = m_matrix_display_vector[row];
		for (neuro_u32 i = 0; i < col_vector.size(); i++)
		{
			const LayerDisplaySetup& layout = col_vector[i];

			COLORREF text_color = RGB(0, 0, 0);

			CBrush* brush;
			if (m_selectedLayout.layout == &layout)
			{
				brush = &m_select_brush;
				text_color = RGB(255, 255, 255);
			}
			else if (m_mouseoverLayout == &layout)
				brush = &m_cur_layer_brush;
			else
				brush = &m_normal_layer_brush;

			dc.SelectObject(brush);

			LOGBRUSH lb;
			brush->GetLogBrush(&lb);
			dc.SetBkColor(lb.lbColor);
			dc.SetTextColor(text_color);

			NP_RECT rcLayout = ViewportToWnd(rc);
			dc.RoundRect(rcLayout.left, rcLayout.top, rcLayout.right, rcLayout.bottom, 10, 10);

			const wchar_t* label = ToString(layout.display.type);
			gui::win32::TextDraw::SingleText(dc, rcLayout, label, gui::win32::horz_align::center);

			rc.pt.y += m_grid_layout.grid_size.height;
		}
		rc.pt.x += m_grid_layout.grid_size.width;
	}

	const _binding_link_vector& binding_link_vector = m_binding_view.GetBindingLinkVector();
	DrawBindingLines(hdc, binding_link_vector, m_binding_view.GetBindingSelectedLink(), m_binding_view.GetBindingMouseoverLink());

	dc.SetTextColor(prev_textcolor);
	dc.SetBkColor(prev_bkcolor);
	dc.SelectObject(pOldBrush);
}

SimDisplaySetupLayerListWnd::_LAYOUT_INFO SimDisplaySetupLayerListWnd::LayoutHitTest(const NP_POINT& point) const
{
	_LAYOUT_INFO ret;
	ret.layout = NULL;

	NP_POINT vp_pt = WndToViewport(point);

	ret.row = vp_pt.x / m_grid_layout.grid_size.width;
	if (ret.row >= m_matrix_display_vector.size())
		return ret;

	ret.col = vp_pt.y / m_grid_layout.grid_size.height;
	if (ret.col >= m_matrix_display_vector[ret.row].size())
		return ret;

	NP_RECT rc;
	rc.left = ret.row*m_grid_layout.grid_size.width + m_grid_layout.item_margin.width;
	rc.right = rc.left + m_grid_layout.item_size.width;
	rc.top = ret.col*m_grid_layout.grid_size.height + m_grid_layout.item_margin.height;
	rc.bottom = rc.top + m_grid_layout.item_size.height;
	if (!rc.PtInRect(vp_pt))
		return ret;

	ret.layout = &const_cast<SimDisplaySetupLayerListWnd*>(this)->m_matrix_display_vector[ret.row][ret.col];
	return ret;
}

void SimDisplaySetupLayerListWnd::ResetSelect()
{
	memset(&m_selectedLayout, 0, sizeof(_LAYOUT_INFO));
}

void SimDisplaySetupLayerListWnd::SelectNeuroUnit(NP_POINT point)
{
	TRACE(L"SelectNeuroUnit. %d, %d\r\n", point.x, point.y);

	ResetSelect();

	m_selectedLayout = LayoutHitTest(point);

	m_binding_view.InitSelection(this);
	{
		const _NEURO_BINDING_LINK* binding_link = NULL;
		if (m_selectedLayout.layout!=NULL)
		{
			m_binding_view.SelectNetworkLayer((network::AbstractLayer*)m_selectedLayout.layout->layer);
		}
		else
		{
			const _binding_link_vector& binding_link_vector = m_binding_view.GetBindingLinkVector();
			binding_link = BindingHitTest(point, binding_link_vector);
		}
		m_binding_view.SetBindingSelectLink(binding_link);
	}
	RefreshDisplay();	// 새로 선택된 link 또는 layer를 다시 그리기 위해
	ShowConfigProperty();
}

void SimDisplaySetupLayerListWnd::MouseLClickEvent(bool bMouseDown, NP_POINT point)
{
	TRACE(_T("MouseLClickEvent : %s\r\n"), bMouseDown ? L"down" : L"up");

	if (bMouseDown)
	{
		/*	직전에 context menu를 출력했고 메뉴 클릭없이 다른곳에 마우스 클릭하면
		WM_LBUTTONDOWN 만 오고 버튼을 놓았을 때 WM_LBUTTONUP가 아닌 WM_MOUSEMOVE 가 온다.
		그렇다고, context menu가 출력됐는지 확인해서 그후에 WM_LBUTTONDOWN이 왔을때 무시하면
		MouseMoveEvent 에서 m_bMouseLButtonDown=false 상태가 되기 때문에 layer 연결이나 멀티 선택을 할수 없다.
		어짜피 마우스 움직였을때도 처리해야 하니까 그냥 냅둔다.
		*/
		m_bMouseLButtonDown = true;

		SelectNeuroUnit(point);
	}
	else if (m_bMouseLButtonDown)	// 이 화면에서 마우스를 누르고 나서 떼었을때.
	{
		m_bMouseLButtonDown = false;
	}
}

void SimDisplaySetupLayerListWnd::MouseRClickEvent(bool bMouseDown, NP_POINT point)
{
	TRACE(_T("MouseRClickEventk : %s\r\n"), bMouseDown ? L"down" : L"up");

	m_bMouseLButtonDown = false;

	if (bMouseDown)
		SelectNeuroUnit(point);
}

void SimDisplaySetupLayerListWnd::MouseMoveEvent(NP_POINT point)
{
	//	TRACE(L"Mouse move\r\n");

	const LayerDisplaySetup* prev_layout = m_mouseoverLayout;;
	m_mouseoverLayout = LayoutHitTest(point).layout;

	{
		const _NEURO_BINDING_LINK* binding_link = NULL;
		if (m_mouseoverLayout == NULL)
		{
			const _binding_link_vector& binding_link_vector = m_binding_view.GetBindingLinkVector();
			binding_link = BindingHitTest(point, binding_link_vector);
		}
		m_binding_view.SetBindingMouseoverLink(binding_link);
	}

	if (m_bMouseLButtonDown)
	{
/*		if (m_selectedLayout != NULL)	// model에서 drag를 시작했을 경우
			BeginDragModel(point, m_selected_unit.model);

		// 모든 dragdrop은 위의 drag.DragDrop 과 RectTracker 에서 시작되고 끝나기 때문에 아래 플래그를 false로 설정해줘야 한다.
		m_bMouseLButtonDown = false;
		*/
	}
	else if (prev_layout != m_mouseoverLayout)
	{
		//		TRACE(_T("mose move : over unit changed\r\n"));
	}
	else
		return;

	RefreshDisplay();
}

void SimDisplaySetupLayerListWnd::ShowConfigProperty()
{
	((SimDisplaySetupWnd&)m_binding_view).ShowConfigProperty(m_selectedLayout.layout);
}

using namespace np::studio;
void SimDisplaySetupLayerListWnd::ContextMenuEvent(NP_POINT point)
{
	SelectNeuroUnit(point);

	std::vector<_menu_item> menuList;
	if (m_selectedLayout.layout == NULL)
		menuList.push_back(studio::_menu_item(_menu::clear_all_displays, IDS_MENU_CLEAR_ALL_DISPLAY_LAYER));
	else
		menuList.push_back(studio::_menu_item(_menu::erase_display, IDS_MENU_DISPLAY_DEL));

	if (menuList.empty())
		return;

	ShowMenu(point, menuList);
}

void SimDisplaySetupLayerListWnd::ProcessMenuCommand(studio::_menu menuID)
{
	if (!GetNSManager())
		return;

	ProviderModelManager& ipd = GetNSManager()->GetProvider();

	switch (menuID)
	{
	case _menu::clear_all_displays:
		ClearAllDisplay();
		break;
	case _menu::erase_display:
		DeleteSelectedDisplay();
		break;
	}
}
