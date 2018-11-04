#include "StdAfx.h"
#include "DesignNetworkWnd.h"
#include "math.h"

#include "NeuroUnitDragSource.h"

#include "project/NeuroStudioProject.h"
#include "util/StringUtil.h"
#include "gui/Win32/GraphicUtil.h"

using namespace np::gui;
using namespace np::studio;

DesignNetworkWnd::DesignNetworkWnd(DeepLearningDesignViewManager& binding_view)
	: AbstractNNWnd(m_network_matrix_modify, { {data_model_cf, sizeof(_DATA_MODEL_DRAG_SOURCE)},{ neuro_layer_cf, sizeof(_LAYER_DRAG_SOURCE) } })
	, NetworkViewManager(binding_view)
{
	m_network_matrix_modify.SetGridLayout({ 0, 20, 0, 0 }, { 120, 60 }, { 30, 25 });

	ResetSelect();

	m_insert_bar_pen.CreatePen(PS_SOLID, 2, RGB(0, 0, 0));
	m_insert_bar_brush.CreateHatchBrush(HS_FDIAGONAL, RGB(0, 0, 0));	// 흠.. 도움이 안됨
}

DesignNetworkWnd::~DesignNetworkWnd()
{
}

void DesignNetworkWnd::LoadView()
{
	ResetSelect();

	m_network_matrix_modify.NetworkChanged(GetNSManager() ? GetNSManager()->GetNetwork() : NULL);

	RefreshScrollBars();
	m_binding_view.MakeBindingLineVector();	// network만 교체하는 경우를 대비해서?? 이럴일이 있을까... 메뉴도 없앴는데

	ShowNetworkProperty();
}

void DesignNetworkWnd::SaveView()
{
	m_network_matrix_modify.UpdateNetwork(*GetNSManager()->GetNetwork());
}

void DesignNetworkWnd::OnScrollChanged()
{
	m_binding_view.MakeBindingLineVector();
}

bool DesignNetworkWnd::GetDataBoundLinePoints(const NP_POINT& from_point, const NeuroBindingModel& model, bool& is_hide, gui::_bezier_pt_vector& points) const
{
	AbstractLayer* layer = (AbstractLayer*)&model;
	MATRIX_POINT mp = m_network_matrix_modify.GetLayerMatrixPoint(*layer);
	if (m_network_matrix_modify.GetLayer(mp) != layer)
		return false;

	is_hide = false;

	CWnd* parent = GetParent();

	NP_RECT rc = m_network_matrix_modify.GetLayerRect(mp);
	rc = ViewportToWnd(rc);

	if (layer->GetLayerType() == _layer_type::input)
	{
		CPoint pt_to(rc.left, rc.top + (rc.bottom - rc.top) / 2);

		CPoint pt1;

		const gui::_GRID_LAYOUT& grid_layout = m_network_matrix_modify.GetGridLayout();
		pt1.x = rc.left - grid_layout.view_margin.left - neuro_float(grid_layout.item_margin.width) * 0.3f;

		CPoint from(from_point.x, from_point.y);
		parent->ClientToScreen(&from);
		ScreenToClient(&from);

		CRect rcClient;
		GetClientRect(rcClient);
		pt1.x -= neuro_float(grid_layout.item_margin.width) * 0.7f * neuro_float(rcClient.Width()- from.x) / neuro_float(rcClient.Width());

		pt1.y = ViewportToWndY(grid_layout.item_margin.height);

		if (pt1.y < 0)
			pt1.y = -10;

		if (pt_to.y < 0)
		{
			is_hide = true;
			pt_to.y = -2;
		}

		ClientToScreen(&pt1);
		parent->ScreenToClient(&pt1);	// 부모좌표로 맞추자
		points.push_back(_BEZIER_POINT({ pt1.x, pt1.y }, false, 1.5f));

		ClientToScreen(&pt_to);
		parent->ScreenToClient(&pt_to);	// 부모좌표로 맞추자
		points.push_back(_BEZIER_POINT({ pt_to.x, pt_to.y }, false, 1.5f, false));
	}
	else
	{
		CPoint pt_to(rc.left + (rc.right - rc.left) / 2, rc.top);
		if (pt_to.y <= 0)
		{
			is_hide = true;
			pt_to.y = -2;
		}

		ClientToScreen(&pt_to);
		parent->ScreenToClient(&pt_to);	// 부모좌표로 맞추자
		points.push_back(_BEZIER_POINT({ pt_to.x, pt_to.y }, false, 1.5f));
	}
	return true;
}

void DesignNetworkWnd::SelectNetworkLayer(network::AbstractLayer* layer)
{
	SelectLayer(layer);
}

void DesignNetworkWnd::Draw(CDC& dc, CRect rcClient)
{
	__super::Draw(dc, rcClient);

	const _binding_link_vector& binding_link_vector = m_binding_view.GetBindingLinkVector();
	DrawBindingLines(dc.GetSafeHdc(), binding_link_vector, m_binding_view.GetBindingSelectedLink(), m_binding_view.GetBindingMouseoverLink());

	DrawDragLine(dc);
}

void DesignNetworkWnd::DrawDragLine(CDC& dc)
{
	if (m_cur_drop_target.dropType == _drop_test::move)	// move 할때는 연결 선만 그린다.
	{
		NP_RECT bar_rc = ViewportToWnd(m_network_matrix_modify.GetGridRect(m_cur_drop_target.pos.matrix_pt));

		enum class _drag_shape { grid, vert, horz };
		_drag_shape drag_shape;
		switch (m_cur_drop_target.pos.pos_in_grid)
		{
		case _POS_INFO_IN_LAYER::_pos_in_grid::side_left:
		case _POS_INFO_IN_LAYER::_pos_in_grid::side_right:
		{
			bar_rc.top = ViewportToWndY(m_network_matrix_modify.GetGridY(0));
			neuro_u32 row_size = m_network_matrix_modify.GetRowCount(m_cur_drop_target.pos.matrix_pt.level);
			bar_rc.bottom = ViewportToWndY(m_network_matrix_modify.GetGridY(row_size));
			if (m_cur_drop_target.pos.pos_in_grid == _POS_INFO_IN_LAYER::_pos_in_grid::side_left)
			{
				bar_rc.left -= 3;
				bar_rc.right = bar_rc.left + 6;
			}
			else
			{
				bar_rc.left += bar_rc.GetWidth() - 3;
				bar_rc.right = bar_rc.left + 6;
			}
			drag_shape = _drag_shape::vert;
			break;
		}
		case _POS_INFO_IN_LAYER::_pos_in_grid::side_up:
			bar_rc.top -= 3;
			bar_rc.bottom = bar_rc.top + 6;
			drag_shape = _drag_shape::horz;
			break;
		case _POS_INFO_IN_LAYER::_pos_in_grid::side_down:
			bar_rc.top += bar_rc.GetHeight() - 3;
			bar_rc.bottom = bar_rc.top + 6;
			drag_shape = _drag_shape::horz;
			break;
		default:
			drag_shape = _drag_shape::grid;
		}

		// drag insert 도 gdi+로 그려보자!
		CBrush* oldBrush = dc.SelectObject(&m_insert_bar_brush);
		CPen* oldPen = dc.SelectObject(&m_insert_bar_pen);

		if (drag_shape == _drag_shape::grid)
		{
			dc.Rectangle(bar_rc.left, bar_rc.top, bar_rc.right, bar_rc.bottom);
		}
		else
		{
			dc.MoveTo(bar_rc.left, bar_rc.top);
			dc.LineTo(bar_rc.left, bar_rc.bottom);
			dc.MoveTo(bar_rc.right, bar_rc.top);
			dc.LineTo(bar_rc.right, bar_rc.bottom);
			if (drag_shape == _drag_shape::horz)
			{
				dc.MoveTo(bar_rc.left, bar_rc.top + bar_rc.GetHeight() / 2);
				dc.LineTo(bar_rc.right, bar_rc.top + bar_rc.GetHeight() / 2);
			}
			else
			{
				dc.MoveTo(bar_rc.left + bar_rc.GetWidth() / 2, bar_rc.top);
				dc.LineTo(bar_rc.left + bar_rc.GetWidth() / 2, bar_rc.bottom);
			}
		}

		dc.SelectObject(oldPen);
		dc.SelectObject(oldBrush);
	}
	else
	{
		NP_POINT drag_start;
		if (m_binding_view.GetDragStartPoint(drag_start))
		{
			CPoint wnd_pt(drag_start.x, drag_start.y);
			ScreenToClient(&wnd_pt);
			gui::win32::GraphicUtil::DrawLine(dc, { wnd_pt.x, wnd_pt.y }, GetCurrentPoint(), RGB(128, 0, 0), 2, 5, 2);
		}
	}
}

void DesignNetworkWnd::ResetSelect()
{
	__super::ClearSelect();

	memset(&m_cur_drop_target, 0, sizeof(_DROP_TARGET_INFO));
	m_cur_drop_target.dropType = _drop_test::none;
}

void DesignNetworkWnd::AfterNetworkSelected(NP_POINT point)
{
	m_binding_view.InitSelection(this);

	const _NEURO_BINDING_LINK* binding_link = NULL;
	if (!m_selected_unit.IsValid())
	{
		const _binding_link_vector& binding_link_vector = m_binding_view.GetBindingLinkVector();
		binding_link = BindingHitTest(point, binding_link_vector);
	}
	m_binding_view.SetBindingSelectLink(binding_link);
}

void DesignNetworkWnd::AfterNetworkMouseMove(NP_POINT point)
{
	const _NEURO_BINDING_LINK* binding_link = NULL;
	if (m_mouseoverLayer == NULL && m_mouseoverLink == NULL)
	{
		const _binding_link_vector& binding_link_vector = m_binding_view.GetBindingLinkVector();
		binding_link = BindingHitTest(point, binding_link_vector);
	}
	m_binding_view.SetBindingMouseoverLink(binding_link);
}

void DesignNetworkWnd::SelectMultiLayers(const NP_2DSHAPE& rc)
{
	__super::SelectMultiLayers(rc);
	ShowNetworkProperty();	// none 출력 해야 하니까

	if (!m_selected_scope.IsValid())
		return;

	std::vector<studio::_menu_item> menuList;
	menuList.push_back(studio::_menu_item(_menu::layer_multi_del, IDS_MENU_LAYER_MULTI_DEL));

	NP_POINT pt = GetCurrentPoint();
	ShowMenu(pt, menuList);	// 현재의 커서위치로 해야 드래그후에 마우스 왼쪽 버튼을 떼었을때의 위치가 된다.
}

void DesignNetworkWnd::BeginDragLayer(NP_POINT pt, const MATRIX_POINT& matrix_pt)
{
	CPoint wnd_pt(pt.x, pt.y);
	ClientToScreen(&wnd_pt);
	m_binding_view.SetDragStartPoint({ wnd_pt.x, wnd_pt.y });

	_LAYER_DRAG_SOURCE source;
	source.mp = matrix_pt;

	NeuroUnitDragDrop drag;
	bool bRet = drag.DragDrop(szLayerClipboardFormat, &source, sizeof(_LAYER_DRAG_SOURCE));

	m_binding_view.SetDragEnd();
}

_drop_test DesignNetworkWnd::DropTest(const _DRAG_SOURCE& source, NP_POINT target_pt)
{
	memset(&m_cur_drop_target, 0, sizeof(_DROP_TARGET_INFO));
	if (!GetNSManager())
		return _drop_test::none;

	_POS_INFO_IN_LAYER to;
	if (!m_network_matrix_modify.LayerHitTest(WndToViewport(target_pt), to))
	{
		m_binding_view.RefreshBindingViews();
		return 	_drop_test::none;
	}

	if (source.cf == neuro_layer_cf)
	{
		if (source.size != sizeof(_LAYER_DRAG_SOURCE))
			return _drop_test::none;

		_LAYER_DRAG_SOURCE layer_drag_source;
		memcpy(&layer_drag_source, source.buffer, sizeof(_LAYER_DRAG_SOURCE));

		if (layer_drag_source.mp == to.matrix_pt)	// 같은 위치는 이동도 연결도 안된다.
			return _drop_test::none;

		if (layer_drag_source.mp.level > to.matrix_pt.level)	// 뒤로만 연결시도 할 수 있다.
			return _drop_test::none;

		if (layer_drag_source.mp.level == to.matrix_pt.level && to.pos_in_grid != _POS_INFO_IN_LAYER::_pos_in_grid::layer)
		{
			// layer 위치를 변경하려고 할때
			if (layer_drag_source.mp.row > to.matrix_pt.row + 1 || layer_drag_source.mp.row + 1 < to.matrix_pt.row)
				m_cur_drop_target.dropType = _drop_test::move;		// 두칸 위아래쪽으로 이동하려고 할때
			else if (layer_drag_source.mp.row == to.matrix_pt.row + 1)
			{	// 한칸 위는 layer 위쪽으로 해야 이동 가능. 한칸 위쪽의 아래는 같은 위치이므로 이동이 무의미
				if (to.pos_in_grid == _POS_INFO_IN_LAYER::_pos_in_grid::side_up)
					m_cur_drop_target.dropType = _drop_test::move;
			}
			else if (layer_drag_source.mp.row + 1 == to.matrix_pt.row)
			{	// 한칸 아래는 layer 아래쪽으로 해야 이동 가능. 한칸 아래쪽의 위는 같은 위치이므로 이동이 무의미
				if (to.pos_in_grid == _POS_INFO_IN_LAYER::_pos_in_grid::side_down)
					m_cur_drop_target.dropType = _drop_test::move;
			}
		}
		else if (to.layer!=NULL)
		{	// 연결 하려고 할때
			if (m_network_matrix_modify.ConnectTest(*GetNSManager()->GetNetwork(), layer_drag_source.mp, to.matrix_pt))
				m_cur_drop_target.dropType = _drop_test::link;
		}
	}
	else if(source.cf == data_model_cf)
	{
		if (source.size != sizeof(_DATA_MODEL_DRAG_SOURCE))
			return _drop_test::none;

		_DATA_MODEL_DRAG_SOURCE data_model_drag_source;
		memcpy(&data_model_drag_source, source.buffer, sizeof(_DATA_MODEL_DRAG_SOURCE));

		if (data_model_drag_source.model->GetModelType() == dp::model::_model_type::producer)
		{
			if(GetNSManager()->GetNetwork()->ConnectTest((dp::model::AbstractProducerModel*) data_model_drag_source.model, to.layer))
				m_cur_drop_target.dropType = _drop_test::link;
		}
	}
	if(m_cur_drop_target.dropType != _drop_test::none)
		m_cur_drop_target.pos = to;

	m_binding_view.RefreshBindingViews();
	return m_cur_drop_target.dropType;
}

bool DesignNetworkWnd::Drop(const _DRAG_SOURCE& source, NP_POINT target_pt)
{
	_drop_test drop_test = DropTest(source, target_pt);
	if(drop_test == _drop_test::none)
		return false;

	DEBUG_OUTPUT(L"");

	bool ret = false;
	if (source.cf == neuro_layer_cf)
	{
		_LAYER_DRAG_SOURCE layer_drag_source;
		memcpy(&layer_drag_source, source.buffer, sizeof(_LAYER_DRAG_SOURCE));

		if (drop_test == _drop_test::move)	// layer 위치를 변경하려고 할때
			ret = m_network_matrix_modify.MoveLayerTo(*GetNSManager()->GetNetwork(), layer_drag_source.mp, m_cur_drop_target.pos.matrix_pt);
		else
			ret = m_network_matrix_modify.Connect(*GetNSManager()->GetNetwork(), layer_drag_source.mp, m_cur_drop_target.pos.matrix_pt);
		if (ret)
			RefreshDisplay();
	}
	else if (source.cf == data_model_cf)
	{
		_DATA_MODEL_DRAG_SOURCE data_model_drag_source;
		memcpy(&data_model_drag_source, source.buffer, sizeof(_DATA_MODEL_DRAG_SOURCE));

		ret = GetNSManager()->GetNetwork()->Connect((dp::model::AbstractProducerModel*) data_model_drag_source.model, m_cur_drop_target.pos.layer);
		if(ret)
			m_binding_view.MakeBindingLineVector();
	}

	return ret;
}

void DesignNetworkWnd::DragLeave()
{
	memset(&m_cur_drop_target, 0, sizeof(_DROP_TARGET_INFO));
	m_cur_drop_target.dropType = _drop_test::none;
}

void DesignNetworkWnd::OnLClickedUnit(bool bMouseDown, NP_POINT point)
{
	if(bMouseDown)
		ShowNetworkProperty();
}

void DesignNetworkWnd::OnRClickedUnit(bool bMouseDown, NP_POINT point)
{
	OnLClickedUnit(bMouseDown, point);
}

bool DesignNetworkWnd::OnContextMenu(NP_POINT point, const _POS_INFO_IN_LAYER& pos_info)
{
	if (!GetNSManager())
		return false;

	ShowNetworkProperty();

	m_insert_layer_pos.Initialize();

	std::vector<studio::_menu_item> menuList;

	if (m_selected_unit.link != NULL)
		menuList.push_back(studio::_menu_item(_menu::link_del, IDS_MENU_LINK_DEL));
	else if (m_binding_view.GetBindingSelectedLink() != NULL)
		menuList.push_back(studio::_menu_item(_menu::link_del, IDS_MENU_LINK_DEL));
	else if (m_selected_unit.layer!=NULL)
	{
		menuList.push_back(studio::_menu_item(_menu::model_del, IDS_MENU_MODEL_DEL));

		if (m_selected_unit.layer->AvailableConnectOutputLayer())
		{
			menuList.push_back(studio::_menu_item(_menu::output_layer_add, IDS_MENU_ADD_OUTPUT_LAYER));

			m_insert_layer_pos.matrix_pt = pos_info.matrix_pt;
			++m_insert_layer_pos.matrix_pt.level;
			m_insert_layer_pos.pos_in_grid = m_network_matrix_modify.GetLayer(m_insert_layer_pos.matrix_pt)==NULL
				? _POS_INFO_IN_LAYER::_pos_in_grid::none : _POS_INFO_IN_LAYER::_pos_in_grid::side_left;
		}
	}
	else if (m_selected_scope.IsValid())
		menuList.push_back(studio::_menu_item(_menu::layer_multi_del, IDS_MENU_LAYER_MULTI_DEL));
	else if (m_network_matrix_modify.AvailableAddLayer(pos_info.matrix_pt))
	{
		m_insert_layer_pos = pos_info;
		menuList.push_back(studio::_menu_item(_menu::layer_add, IDS_MENU_LAYER_ADD));
	}

	if (menuList.empty())
		return false;

	ShowMenu(point, menuList);
	return true;
}

void DesignNetworkWnd::ProcessMenuCommand(studio::_menu menuID)
{
	if (!GetNSManager())
		return;

	switch (menuID)
	{
	case _menu::layer_multi_del:
	case _menu::model_del:
	{
		_std_u32_vector deleted_uid_vector;
		if (menuID == _menu::layer_multi_del)
		{
			if (!m_network_matrix_modify.DeleteLayers(*GetNSManager()->GetNetwork(), m_selected_scope, deleted_uid_vector))
				return;
		}
		else
		{
			if (!m_network_matrix_modify.DeleteLayer(*GetNSManager()->GetNetwork(), m_selected_unit.layer, deleted_uid_vector))
				return;
		}
		// 여기에서 display 정보를 지우지 않으면 나중에 uid를 재사용하게 되므로 문제가 생긴다.
		project::_layer_display_info_map& layer_display_info_map = GetProject()->GetSimManager().GetLayerDisplayInfoMap();
		for (neuro_u32 i = 0; i < deleted_uid_vector.size(); i++)
			layer_display_info_map.erase(deleted_uid_vector[i]);

		ResetSelect();
		RefreshScrollBars();

		DEBUG_OUTPUT(L"");
		m_binding_view.MakeBindingLineVector();
		return;
	}
	case _menu::layer_add:
		if (!AddLayer(false))
			return;

		RefreshScrollBars();

		DEBUG_OUTPUT(L"");
		m_binding_view.MakeBindingLineVector();	// 중간 삽입으로 뒤에 있을 output layer의 binding 위치가 바뀔수 있으므로
		return;
	case _menu::output_layer_add:
		if (!AddLayer(true))
			return;
		RefreshScrollBars();
		return;
	case _menu::link_del:
		if (m_selected_unit.link != NULL && m_selected_unit.link->HasLink())
		{
			if (!m_network_matrix_modify.DisConnect(*GetNSManager()->GetNetwork(), m_selected_unit.link->from, m_selected_unit.link->to))
				return;
			ResetSelect();
		}
		else if (m_binding_view.GetBindingSelectedLink() != NULL)
		{
			NetworkBindingModel* binding_to =(NetworkBindingModel*) m_binding_view.GetBindingSelectedLink()->to;
			binding_to->RemoveBinding((NetworkBindingModel*)m_binding_view.GetBindingSelectedLink()->from);

			DEBUG_OUTPUT(L"");
			m_binding_view.MakeBindingLineVector();
			return;
		}
	}
}

// layer의 type 또는 속성들을 변경하려면 이것을 통해서!
bool DesignNetworkWnd::ChangeLayerType(HiddenLayer* layer, _layer_type layer_type, const nsas::_LAYER_STRUCTURE_UNION* entry, _slice_input_vector* org_erased_input_vector)
{
	_slice_input_vector erased_input_vector;
	_slice_input_vector* erased = org_erased_input_vector ? org_erased_input_vector : &erased_input_vector;

	if (!layer->ChangeLayerType(layer_type, entry, erased))
		return false;

	m_network_matrix_modify.DisconnectedInput(layer->uid, *erased);
	RefreshDisplay();

	return true;
}

bool DesignNetworkWnd::AddLayer(bool is_output)
{
	if (!GetNSManager())
		return false;

	network::_layer_type desire_hidden_type = m_last_set_entries.m_hidden_type;
	network::AbstractLayer* layer = NULL;
	if (is_output)
	{
		if (m_selected_unit.layer == NULL || !m_selected_unit.layer->AvailableConnectOutputLayer())
			return false;

		desire_hidden_type = _layer_type::output;
	}

	layer = m_network_matrix_modify.AddLayer(*GetNSManager()->GetNetwork(), desire_hidden_type, m_insert_layer_pos.matrix_pt, m_insert_layer_pos.pos_in_grid);
	if (!layer)
		return false;
	if (layer->GetLayerType() != network::_layer_type::input)
	{
		((network::HiddenLayer*)layer)->ChangeLayerType(layer->GetLayerType(), &m_last_set_entries.GetEntry(layer->GetLayerType()));
		((network::HiddenLayer*)layer)->SetActivation(m_last_set_entries.m_activation);
	}

	if (is_output)
	{
		MATRIX_POINT from = m_network_matrix_modify.GetLayerMatrixPoint(*m_selected_unit.layer);
		MATRIX_POINT to = m_network_matrix_modify.GetLayerMatrixPoint(*layer);
		m_network_matrix_modify.Connect(*GetNSManager()->GetNetwork(), from, to);
	}

	m_selected_unit.Initialize();
	m_selected_unit.layer = layer;

	ShowNetworkProperty();
	return true;
}

#include "DeeplearningDesignView.h"
void DesignNetworkWnd::ShowNetworkProperty()
{
	if (!GetNSManager())
		return;

	ModelPropertyWnd& property_view = ((DeeplearningDesignView&)m_binding_view).GetPropertyPane();

	if (m_selected_unit.layer == NULL)
		property_view.SetModelProperty(*this, GetNSManager()->GetNetwork());
	else
		property_view.SetModelProperty(*this, m_last_set_entries, m_selected_unit.layer);
}
