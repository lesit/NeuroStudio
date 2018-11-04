#include "stdafx.h"

#include "DesignPreprocessorWnd.h"
#include "gui/Win32/TextDraw.h"

#include "NeuroUnitDragSource.h"

DesignPreprocessorWnd::DesignPreprocessorWnd(DeepLearningDesignViewManager& binding_view)
	: CMappingWnd({ { data_model_cf, sizeof(_DATA_MODEL_DRAG_SOURCE) },{ neuro_layer_cf, sizeof(_LAYER_DRAG_SOURCE) } })
	, DataViewManager(binding_view)
{
	m_grid_layout.SetLayout({ 10, 10, 10, 20 }, { 120, 50 }, { 20, 20 });
//	m_source_desc_height = 70;
	m_source_desc_height = 0;

	ClearAll();
	m_mouseoverModel = NULL;
	m_mouseoverLink = NULL;
	m_bMouseLButtonDown = false;
}

DesignPreprocessorWnd::~DesignPreprocessorWnd()
{

}

void DesignPreprocessorWnd::ClearAll()
{
	m_seperate_vertical_line_x = -1;
	m_max_width = 0;
	m_max_height = m_source_desc_height + 2 * m_grid_layout.grid_size.height;	// 최소 높이

	m_predict_layout_vector.clear();
	m_learn_layout_vector.clear();
	m_link_vector.clear();

	m_producer_scope_rect = { 0,0,0,0 };
	m_insert_point.Set(0, 0);

	memset(&m_cur_drop_target, 0, sizeof(_DROP_TARGET_INFO));
}

void DesignPreprocessorWnd::LoadView()
{
	ResetSelect();

	CompositeAll();
}

void DesignPreprocessorWnd::SaveView()
{
	if (!GetNSManager())
		return;
}

void DesignPreprocessorWnd::SetIntegratedLayout(bool is_integrated)
{
	bool old_integrated = GetNSManager()->GetProvider().GetLearnProvider()==NULL;
	if(old_integrated != is_integrated)
	{
		GetNSManager()->GetProvider().IntegratedProvider(is_integrated);
		m_learn_layout_vector.clear();

		RefreshScrollBars();

		DEBUG_OUTPUT(L"");
		m_binding_view.MakeBindingLineVector();
	}
}

NP_SIZE DesignPreprocessorWnd::GetScrollTotalViewSize() const
{
	return NP_SIZE(m_max_width, m_max_height);
}

neuro_u32 DesignPreprocessorWnd::GetScrollMoving(bool is_horz) const
{
	return is_horz ? m_grid_layout.grid_size.width / 5 : m_grid_layout.grid_size.height / 5;
}

void DesignPreprocessorWnd::CompositeAll()
{
	ClearAll();

	if (!GetNSManager())
		return;

	const ProviderModelManager& ipd = GetNSManager()->GetProvider();

	_reader_level_vector predict_reader_level_vector;
	CompositeReaderLevelVector(ipd.GetPredictProvider().GetReaderVector(), predict_reader_level_vector);

	_reader_level_vector learn_reader_level_vector;
	if(ipd.GetLearnProvider())
		CompositeReaderLevelVector(ipd.GetLearnProvider()->GetReaderVector(), learn_reader_level_vector);

	neuro_u32 max_level_count = max(predict_reader_level_vector.size(), learn_reader_level_vector.size()) + 1;	// + 1 은 producer를 위한 것
	if (max_level_count < 2)	// reader, producer 등 최소 두개가 있어야  한다.
		max_level_count = 2;

	NP_POINT end_pt;
	NP_POINT start_pt(0, m_source_desc_height);
	CompositeModelVector(ipd.GetPredictProvider().GetProducerVector(), predict_reader_level_vector
			, 0, max_level_count, m_predict_layout_vector, end_pt);

	m_max_height = end_pt.y;

	if (ipd.GetLearnProvider())
	{
		m_seperate_vertical_line_x = end_pt.x + 10;// 간격을 더 띄운다.

		CompositeModelVector(ipd.GetLearnProvider()->GetProducerVector(), learn_reader_level_vector
			, m_seperate_vertical_line_x +  10, max_level_count, m_learn_layout_vector, end_pt);

		m_max_width = end_pt.x;
		m_max_height = max(m_max_height, end_pt.y);
	}

	m_producer_scope_rect.left = 0;
	m_producer_scope_rect.right = m_max_width;
	m_producer_scope_rect.top = m_max_height - m_grid_layout.grid_size.height;
	m_producer_scope_rect.bottom = m_max_height;

	DEBUG_OUTPUT(L"");
	m_binding_view.MakeBindingLineVector();

	GetParent()->SendMessage(WM_SIZE);
}

void DesignPreprocessorWnd::CompositeModelVector(const _producer_model_vector& producer_vector, const _reader_level_vector& reader_level_vector
											, long start_x, neuro_u32 level_count
											, _layout_level_vector& layout_level_vector, NP_POINT& max_end_pt)
{
	if (level_count == 0)
		return;

	typedef std::unordered_map<const AbstractPreprocessorModel*, std::vector<_MODEL_LAYOUT*>> _model_child_map;
	_model_child_map model_child_map;

	NP_2DSHAPE rcModel;
	rcModel.pt.x = start_x + m_grid_layout.item_margin.width;
	rcModel.pt.y = m_source_desc_height + (level_count - 1) * m_grid_layout.grid_size.height + m_grid_layout.item_margin.height;
	rcModel.sz = m_grid_layout.item_size;

	max_end_pt.x = start_x + max(producer_vector.size(), 1) * m_grid_layout.grid_size.width;	// 최소 한개 자리를 만든다.
	max_end_pt.y = rcModel.Bottom() + m_grid_layout.item_margin.height;

	if (producer_vector.size() == 0)
		return;

	// 아래부터 위로 그려질 위치를 결정한다. 즉, prdocuer부터 맨 위 reader 까지
	layout_level_vector.resize(1 + reader_level_vector.size());

	{
		_model_layout_vector& model_layout_vector = layout_level_vector[0];
		model_layout_vector.resize(producer_vector.size());
		for (neuro_u32 i = 0; i < producer_vector.size(); i++)
		{
			_MODEL_LAYOUT& layout = model_layout_vector[i];

			layout.rc = rcModel;
			layout.model = producer_vector[i];

			rcModel.pt.x += m_grid_layout.grid_size.width;

			if (layout.model->GetInput())
			{	// 부모 즉 input의 child vector에 추가시킨다.
				std::vector<_MODEL_LAYOUT*>& child_vector = model_child_map[layout.model->GetInput()];
				child_vector.push_back(&layout);
			}
		}
	}

	_reader_level_vector::const_reverse_iterator it = reader_level_vector.rbegin();
	for (neuro_32 level = 1; it != reader_level_vector.rend(); it++, level++)
	{
		const std::vector<AbstractReaderModel*>& row_vector = *it;

		_model_layout_vector& model_layout_vector = layout_level_vector[level];

		for (neuro_u32 i = 0; i < row_vector.size(); i++)
		{
			AbstractReaderModel* model = row_vector[i];

			_model_child_map::const_iterator it_child = model_child_map.find(model);
			if (it_child == model_child_map.end())
				continue;

			const std::vector<_MODEL_LAYOUT*>& child_vector = it_child->second;

			model_layout_vector.resize(model_layout_vector.size() + 1);
			_MODEL_LAYOUT& layout = model_layout_vector.back();

			layout.model = model;

			long start = child_vector[0]->rc.pt.x;
			long end = child_vector[child_vector.size() - 1]->rc.Right();
			layout.rc.pt.x = start + (end - start) / 2 - m_grid_layout.item_size.width / 2;
			layout.rc.pt.y = child_vector[0]->rc.pt.y - m_grid_layout.grid_size.height;
			layout.rc.sz = m_grid_layout.item_size;

			neuro_u32 link_index = m_link_vector.size();
			m_link_vector.resize(link_index + child_vector.size());
			for (neuro_u32 child = 0; child < child_vector.size(); child++, link_index++)
			{
				const _MODEL_LAYOUT* output = child_vector[child];
				_LINK_INFO& link = m_link_vector[link_index];
				link.from = layout.model;
				link.to = output->model;

				link.line.bezier.start = { layout.rc.pt.x + layout.rc.sz.width / 2, layout.rc.pt.y + layout.rc.sz.height};
				NP_POINT ptTo(output->rc.pt.x + output->rc.sz.width / 2, output->rc.pt.y );
				link.line.draw_type = _line_draw_type::bezier;
				link.line.arrow_type = _line_arrow_type::end;
				link.line.bezier.points.push_back(_BEZIER_POINT(ptTo, false, 1.f));
			}
			if (model->GetInput())
			{	// 부모 즉 input의 child vector에 추가시킨다.
				std::vector<_MODEL_LAYOUT*>& child_vector = model_child_map[model->GetInput()];
				child_vector.push_back(&layout);
			}
		}

		if (model_layout_vector.size() == 0)
		{
			layout_level_vector.erase(layout_level_vector.begin() + level, layout_level_vector.end());
			return;
		}
	}
}

void DesignPreprocessorWnd::CompositeReaderLevelVector(const _reader_model_vector& reader_vector, _reader_level_vector& reader_level_vector)
{
	typedef std::unordered_map<const AbstractPreprocessorModel*, neuro_u32> _model_depth_map;
	_model_depth_map model_depth_map;

	neuro_u32 level = 0;
	for (neuro_u32 i = 0; i < reader_vector.size(); i++)
	{
		AbstractReaderModel* model = reader_vector[i];

		if (model->GetInput() != NULL)
		{
			_model_depth_map::iterator it_parent = model_depth_map.find(model->GetInput());
			if (it_parent != model_depth_map.end())	// 입력과 같은 level이면 다음 level로 변경
			{
				level = it_parent->second + 1;
			}
		}
		model_depth_map[model] = level;

		reader_level_vector.resize(level + 1);
		reader_level_vector[level].push_back(model);
	}
}

void DesignPreprocessorWnd::GetBindedModelVector(_binding_source_vector& model_vector) const
{
	if (!GetNSManager())
		return;

	const ProviderModelManager& ipd = GetNSManager()->GetProvider();

	CWnd* parent = GetParent();

	auto get_binding_vector = [&](const _layout_level_vector& layout_level_vector)
	{
		if (layout_level_vector.size() == 0)
			return;

		const _model_layout_vector& layout_vector = layout_level_vector[0];
		for (neuro_u32 i = 0; i < layout_vector.size(); i++)
		{
			AbstractProducerModel* model = (AbstractProducerModel*)layout_vector[i].model;
			if (model->GetBindingSet().size()>0)
			{
				NP_RECT rcModel = ViewportToWnd(layout_vector[i].rc);

				CPoint wnd_pt(rcModel.left + (rcModel.right - rcModel.left) / 2, rcModel.bottom);
				ClientToScreen(&wnd_pt);
				parent->ScreenToClient(&wnd_pt);	// 부모좌표로 맞추자

				const _neuro_binding_model_set& binding_model_set = model->GetBindingSet();

				neuro_u32 start = model_vector.size();
				model_vector.resize(model_vector.size() + binding_model_set.size());
				for (_neuro_binding_model_set::const_iterator it = binding_model_set.begin(); it != binding_model_set.end(); it++)
				{
					_BINDING_SOURCE_MODEL& binding = model_vector[start++];
					binding.from_point = { wnd_pt.x, wnd_pt.y };
					binding.from = model;
					binding.to = *it;
				}
			}
		}
	};
	get_binding_vector(m_predict_layout_vector);
	get_binding_vector(m_learn_layout_vector);
}

void DesignPreprocessorWnd::OnScrollChanged()
{
	m_binding_view.MakeBindingLineVector();
}

#include "desc/PreprocessorDesc.h"
void DesignPreprocessorWnd::Draw(CDC& dc, CRect rcClient)
{
	HDC hdc = dc.GetSafeHdc();
	if(m_source_desc_height>0)
	{
		Gdiplus::Pen pen(Gdiplus::Color(0, 0, 0), 1.5f);
		gui::win32::GraphicUtil::CompositeLinePen(pen, gui::_line_arrow_type::none, gui::_line_dash_type::dot);
		gui::win32::GraphicUtil::DrawLine(hdc
			, { (neuro_32)rcClient.left, (neuro_32) (rcClient.top + m_source_desc_height) }
			, { (neuro_32)rcClient.right, (neuro_32) (rcClient.top + m_source_desc_height) }
			, pen);
	}
	
	for (neuro_u32 i = 0; i < m_link_vector.size(); i++)
	{
		const _LINK_INFO& link = m_link_vector[i];

		NP_POINT pt_from = ViewportToWnd(link.line.bezier.start);
		NP_POINT pt_to = ViewportToWnd(link.line.bezier.points[link.line.bezier.points.size() - 1].pt);

		// 연결선이 화면에 조금이라도 보이는 것만 그린다.
		if (pt_from.x > rcClient.right || pt_to.x < rcClient.left)
			continue;

		if (pt_from.y < rcClient.top && pt_to.y < rcClient.top ||
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

	auto draw_models = [&](const _layout_level_vector& layout_level_vector)
	{
		_layout_level_vector::const_iterator it = layout_level_vector.begin();
		for (; it != layout_level_vector.end(); it++)
		{
			const _model_layout_vector& layout_vector = *it;
			for (neuro_u32 i = 0; i < layout_vector.size(); i++)
			{
				const _MODEL_LAYOUT& layout = layout_vector[i];

				COLORREF text_color = RGB(54, 65, 52);

				CBrush* brush;
				if (m_selected_unit.model == layout.model)
				{
					brush = &m_select_brush;
					text_color = RGB(255, 255, 255);
				}
				else if (m_mouseoverModel == layout.model)
					brush = &m_cur_layer_brush;
				else
					brush = &m_normal_layer_brush;

				dc.SelectObject(brush);

				LOGBRUSH lb;
				brush->GetLogBrush(&lb);
				dc.SetBkColor(lb.lbColor);
				dc.SetTextColor(text_color);

				NP_RECT rcModel = ViewportToWnd(layout.rc);
				dc.RoundRect(rcModel.left, rcModel.top, rcModel.right, rcModel.bottom, 10, 10);

				const wchar_t* label = str_rc::PreprocessorDesc::GetName(*layout.model);
				gui::win32::TextDraw::SingleText(dc, rcModel, label, gui::win32::horz_align::center);
			}
		}
	};

	CBrush* pOldBrush = dc.GetCurrentBrush();
	COLORREF prev_textcolor = dc.GetTextColor();
	COLORREF prev_bkcolor = dc.GetBkColor();

	draw_models(m_predict_layout_vector);

	if(m_seperate_vertical_line_x)
	{
		Gdiplus::Pen pen(Gdiplus::Color(0, 0, 0), 1.5f);
		gui::win32::GraphicUtil::CompositeLinePen(pen, gui::_line_arrow_type::none, gui::_line_dash_type::dash);

		gui::win32::GraphicUtil::DrawLine(hdc
			, { m_seperate_vertical_line_x, rcClient.top + (neuro_32)m_source_desc_height + 5 }
			, { m_seperate_vertical_line_x, rcClient.bottom - 5 }
			, pen);

		draw_models(m_learn_layout_vector);
	}

	const _binding_link_vector& binding_link_vector = m_binding_view.GetBindingLinkVector();
	DrawBindingLines(hdc, binding_link_vector, m_binding_view.GetBindingSelectedLink(), m_binding_view.GetBindingMouseoverLink());

	NP_POINT drag_start;
	if (m_binding_view.GetDragStartPoint(drag_start))
	{
		CPoint wnd_pt(drag_start.x, drag_start.y);
		ScreenToClient(&wnd_pt);
		gui::win32::GraphicUtil::DrawLine(hdc, { wnd_pt.x, wnd_pt.y }, GetCurrentPoint(), RGB(128, 0, 0), 2, 5, 2);
	}

	dc.SetTextColor(prev_textcolor);
	dc.SetBkColor(prev_bkcolor);
	dc.SelectObject(pOldBrush);
}

const DesignPreprocessorWnd::_LINK_INFO* DesignPreprocessorWnd::LinkHitTest(const NP_POINT& point) const
{
	NP_POINT vp_pt = WndToViewport(point);

	for (neuro_u32 i = 0; i < m_link_vector.size(); i++)
	{
		const _LINK_INFO& link = m_link_vector[i];
		if (LineHitTest(vp_pt, link.line))
			return &link;
	}

	return false;
}

#include <functional>
AbstractPreprocessorModel* DesignPreprocessorWnd::ModelHitTest(const NP_POINT& point) const
{
	NP_POINT vp_pt = WndToViewport(point);

	std::function<AbstractPreprocessorModel*(const _layout_level_vector& layout_level_vector)> model_hittest;
	model_hittest = [&](const _layout_level_vector& layout_level_vector) ->AbstractPreprocessorModel*
	{
		_layout_level_vector::const_iterator it = layout_level_vector.begin();
		for (; it != layout_level_vector.end(); it++)
		{
			const _model_layout_vector& layout_vector = *it;
			for (neuro_u32 i = 0; i < layout_vector.size(); i++)
			{
				const _MODEL_LAYOUT& layout = layout_vector[i];

				if (layout.rc.PtInRect(vp_pt))
					return layout.model;
			}
		}
		return NULL;
	};
	AbstractPreprocessorModel* hit = model_hittest(m_predict_layout_vector);
	if (hit)
		return hit;
	return model_hittest(m_learn_layout_vector);
}

void DesignPreprocessorWnd::ResetSelect()
{
	m_selected_unit.Initialize();

	memset(&m_cur_drop_target, 0, sizeof(_DROP_TARGET_INFO));
}

void DesignPreprocessorWnd::SelectNeuroUnit(NP_POINT point)
{
	TRACE(L"SelectNeuroUnit. %d, %d\r\n", point.x, point.y);

	m_binding_view.InitSelection(this);

	ResetSelect();

	const _LINK_INFO* link = LinkHitTest(point);
	if (link != NULL)
		m_selected_unit.link = link;
	else
		m_selected_unit.model = ModelHitTest(point);

	{
		const _NEURO_BINDING_LINK* binding_link = NULL;
		if (!m_selected_unit.IsValid())
		{
			const _binding_link_vector& binding_link_vector = m_binding_view.GetBindingLinkVector();
			binding_link = BindingHitTest(point, binding_link_vector);
		}
		m_binding_view.SetBindingSelectLink(binding_link);
	}
	ShowConfigProperty();
}

void DesignPreprocessorWnd::MouseLClickEvent(bool bMouseDown, NP_POINT point)
{
	TRACE(_T("MouseLClickEvent : %s\r\n"), bMouseDown ? L"down" : L"up");

	SelectNeuroUnit(point);
	RefreshDisplay();	// 새로 선택된 link 또는 layer를 다시 그리기 위해

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
			TRACE(L"MouseLClickEvent : not focused\r\n");
	}
	else if (m_bMouseLButtonDown)	// 이 화면에서 마우스를 누르고 나서 떼었을때.
	{
		m_bMouseLButtonDown = false;
	}
}

void DesignPreprocessorWnd::MouseRClickEvent(bool bMouseDown, NP_POINT point)
{
	TRACE(_T("MouseRClickEventk : %s\r\n"), bMouseDown ? L"down" : L"up");

	m_bMouseLButtonDown = false;

	SelectNeuroUnit(point);
	RefreshDisplay();	// 새로 선택된 link 또는 layer를 다시 그리기 위해
}

void DesignPreprocessorWnd::MouseMoveEvent(NP_POINT point)
{
	//	TRACE(L"Mouse move\r\n");

	const AbstractPreprocessorModel* prev_model = m_mouseoverModel;;
	m_mouseoverModel = ModelHitTest(point);

	const _LINK_INFO* prev_Link = m_mouseoverLink;
	m_mouseoverLink = LinkHitTest(point);

	{
		const _NEURO_BINDING_LINK* binding_link = NULL;
		if (m_mouseoverModel == NULL && m_mouseoverLink == NULL)
		{
			const _binding_link_vector& binding_link_vector = m_binding_view.GetBindingLinkVector();
			binding_link = BindingHitTest(point, binding_link_vector);
		}
		m_binding_view.SetBindingMouseoverLink(binding_link);
	}

	if (m_bMouseLButtonDown)
	{
		if (m_selected_unit.model!=NULL)	// model에서 drag를 시작했을 경우
			BeginDragModel(point, m_selected_unit.model);

		// 모든 dragdrop은 위의 drag.DragDrop 과 RectTracker 에서 시작되고 끝나기 때문에 아래 플래그를 false로 설정해줘야 한다.
		m_bMouseLButtonDown = false;
	}
	else if (prev_model != m_mouseoverModel)
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

#include "DeeplearningDesignView.h"
void DesignPreprocessorWnd::ShowConfigProperty()
{
	ModelPropertyWnd& property_view = ((DeeplearningDesignView&)m_binding_view).GetPropertyPane();

	if (m_selected_unit.model == NULL)
		property_view.Clear();
	else
		property_view.SetModelProperty(*this, (AbstractReaderModel*)m_selected_unit.model);

	SetFocus();
}

bool DesignPreprocessorWnd::ReplacePreprocessorModel(AbstractPreprocessorModel* old_model, AbstractPreprocessorModel* new_model)
{
	DataProviderModel& provider = old_model->GetProvider();

	if(!provider.ReplacePreprocessorModel(old_model, new_model))
		return false;

	bool prev_selected = m_selected_unit.model == old_model;

	LoadView();

	if(prev_selected)
		m_selected_unit.model = new_model;

	// 흠.. 이때 Refresh 를 하지 않아도 선택된게 보인다는게 신기...
	return true;
}

using namespace np::studio;
void DesignPreprocessorWnd::ContextMenuEvent(NP_POINT point)
{
	SelectNeuroUnit(point);
	RefreshDisplay();	// 새로 선택된 link 또는 layer를 다시 그리기 위해

	std::vector<_menu_item> menuList;
	if (m_selected_unit.link != NULL && m_selected_unit.link->HasLink())
		menuList.push_back(studio::_menu_item(_menu::link_del, IDS_MENU_LINK_DEL));
	else if (m_binding_view.GetBindingSelectedLink() != NULL)
		menuList.push_back(studio::_menu_item(_menu::link_del, IDS_MENU_LINK_DEL));
	else if (m_selected_unit.model != NULL)
	{
		menuList.push_back(studio::_menu_item(_menu::model_del, IDS_MENU_MODEL_DEL));

		std::vector<dp::model::_reader_type> in_types = m_selected_unit.model->GetAvailableInputReaderTypeVector();
		for (neuro_u32 i = 0; i < in_types.size(); i++)
		{
			switch (in_types[i])
			{
			case _reader_type::text:
				menuList.push_back(studio::_menu_item(_menu::bin_reader_add_to_input, IDS_MENU_BIN_READER_ADD_TO_INPUT));
				break;
			case _reader_type::binary:
				menuList.push_back(studio::_menu_item(_menu::text_reader_add_to_input, IDS_MENU_TEXT_READER_ADD_TO_INPUT));
				break;
			}
		}
	}
	else
	{
		NP_RECT rc = ViewportToWnd(m_producer_scope_rect);
		if (point.y >= rc.top && point.y < rc.bottom)
		{
			m_insert_point = point;
			menuList.push_back(studio::_menu_item(_menu::producer_add, IDS_MENU_PRODUCER_ADD));
		}
	}

	if (menuList.empty())
		return;

	ShowMenu(point, menuList);
}

void DesignPreprocessorWnd::ProcessMenuCommand(studio::_menu menuID)
{
	if (!GetNSManager())
		return;

	ProviderModelManager& ipd = GetNSManager()->GetProvider();

	switch (menuID)
	{
	case _menu::model_del:
		if(!ipd.DeleteDataModel(m_selected_unit.model))
			return;

		ResetSelect();
		break;
	case _menu::bin_reader_add_to_input:
	case _menu::text_reader_add_to_input:
	{
		_reader_type type = _reader_type::unknown;
		switch (menuID)
		{
		case _menu::bin_reader_add_to_input:
			type = _reader_type::binary;
			break;
		case _menu::text_reader_add_to_input:
			type = _reader_type::text;
			break;
		}
		if (type == _reader_type::unknown)
			return;

		AbstractReaderModel* reader = ipd.AddReaderModel(*m_selected_unit.model, type);
		if (reader==NULL)
			return;

		ResetSelect();
		m_selected_unit.model = reader;
		break;
	}
	case _menu::producer_add:
	{
		DataProviderModel* provider = m_insert_point.x < m_seperate_vertical_line_x ? &ipd.GetPredictProvider() : ipd.GetLearnProvider();
		if (provider)
		{
			AbstractProducerModel* producer = provider->AddProducerModel(np::dp::model::_producer_type::numeric);

			ResetSelect();
			m_selected_unit.model = producer;
		}
		break;
	}
	case _menu::link_del:
		if (m_selected_unit.link != NULL && m_selected_unit.link->HasLink())
		{
			if (!ipd.Disconnect((AbstractReaderModel*)m_selected_unit.link->from, m_selected_unit.link->to))
				return;

			ResetSelect();
		}
		else if (m_binding_view.GetBindingSelectedLink() != NULL)
		{
			NetworkBindingModel* binding_to = (NetworkBindingModel*)m_binding_view.GetBindingSelectedLink()->to;
			binding_to->RemoveBinding((NetworkBindingModel*)m_binding_view.GetBindingSelectedLink()->from);
			m_binding_view.SetBindingSelectLink(NULL);
		}
		break;
	}

	CompositeAll();
	RefreshScrollBars();
	m_binding_view.MakeBindingLineVector();
}

void DesignPreprocessorWnd::BeginDragModel(NP_POINT pt, AbstractPreprocessorModel* model)
{
	CPoint wnd_pt(pt.x, pt.y);
	ClientToScreen(&wnd_pt);
	m_binding_view.SetDragStartPoint({ wnd_pt.x, wnd_pt.y });

	_DATA_MODEL_DRAG_SOURCE source;
	source.model = model;

//	SetCapture();

	NeuroUnitDragDrop drag;
	bool bRet = drag.DragDrop(szDataModelClipboardFormat, &source, sizeof(_DATA_MODEL_DRAG_SOURCE));

//	ReleaseCapture();
	m_binding_view.SetDragEnd();
}

_drop_test DesignPreprocessorWnd::DropTest(const _DRAG_SOURCE& source, NP_POINT target_pt)
{
	memset(&m_cur_drop_target, 0, sizeof(_DROP_TARGET_INFO));
	if (!GetNSManager())
		return _drop_test::none;

	m_cur_drop_target.model = ModelHitTest(target_pt);
	if (m_cur_drop_target.model == NULL)
	{
		m_binding_view.RefreshBindingViews();
		return _drop_test::none;
	}

	if (source.cf == data_model_cf)
	{
		if (source.size != sizeof(_DATA_MODEL_DRAG_SOURCE))
			return _drop_test::none;

		_DATA_MODEL_DRAG_SOURCE data_model_drag_source;
		memcpy(&data_model_drag_source, source.buffer, sizeof(_DATA_MODEL_DRAG_SOURCE));
		if (data_model_drag_source.model->GetModelType() == dp::model::_model_type::reader)
		{
			if(m_cur_drop_target.model->AvailableInput(((AbstractReaderModel*)data_model_drag_source.model)->GetReaderType()))
				m_cur_drop_target.dropType = _drop_test::link;
		}
	}
	else if (source.cf == neuro_layer_cf)	// producer -> in/out layer로 가야 한다.
	{
	}

	m_binding_view.RefreshBindingViews();

	return m_cur_drop_target.dropType;
}

bool DesignPreprocessorWnd::Drop(const _DRAG_SOURCE& source, NP_POINT target_pt)
{
	_drop_test drop_test = DropTest(source, target_pt);
	if (drop_test == _drop_test::none)
		return false;

	_DATA_MODEL_DRAG_SOURCE model_drag_source;
	memcpy(&model_drag_source, source.buffer, sizeof(_DATA_MODEL_DRAG_SOURCE));

	m_cur_drop_target.model->SetInput((AbstractReaderModel*)model_drag_source.model);
	RefreshDisplay();
	return true;
}

void DesignPreprocessorWnd::DragLeave()
{
	memset(&m_cur_drop_target, 0, sizeof(_DROP_TARGET_INFO));
}

/*	중간 중간에 비어 있는 경우를 위해 구현된 것
void DesignPreprocessorWnd::CompositeModelVector(const DataProviderModel& provider, long start_x, _layout_level_vector& layout_level_vector, long& last_x)
{
	last_x = 0;

	// reader tree를 만들어야 한다.
	_reader_level_vector reader_level_vector;
	CompositeReaderLevelVector(provider.GetReaderVector(), reader_level_vector);

	neuro_32 depth = reader_level_vector.size();

	typedef std::unordered_map<const AbstractPreprocessorModel*, std::vector<_MODEL_LAYOUT*>> _model_child_map;
	_model_child_map model_child_map;

	layout_level_vector.resize(depth + 1);

	NP_2DSHAPE rc;
	rc.pt.x = start_x;
	rc.pt.y = min(depth, 1) * m_grid_layout.grid_size.height;
	rc.sz = m_grid_layout.grid_size;

	// 아래부터 위로 그려질 위치를 결정한다. 즉, prdocuer부터 맨 위 reader 까지
	_model_layout_vector& model_layout_vector = layout_level_vector[depth];
	const _producer_model_vector& producer_vector = provider.GetProducerVector();
	for (neuro_u32 i = 0; i < producer_vector.size(); i++)
	{
		model_layout_vector.resize(model_layout_vector.size() + 1);
		_MODEL_LAYOUT& layout = model_layout_vector.back();

		layout.grid_rc = rc;
		layout.model = producer_vector[i];

		rc.pt.x += m_grid_layout.grid_size.width;

		if (layout.model->GetInput())
		{	// 부모 즉 input의 child vector에 추가시킨다.
			std::vector<_MODEL_LAYOUT*>& child_vector = model_child_map[layout.model->GetInput()];
			child_vector.push_back(&layout);
		}

		layout.right_layout = NULL;
		if (model_layout_vector.size() > 1)
			model_layout_vector[model_layout_vector.size() - 2].right_layout = &layout;
	}
	if (model_layout_vector.size() > 0)
		last_x = min(last_x, model_layout_vector.back().grid_rc.Right());

	depth = reader_level_vector.size() - 1;
	_reader_level_vector::reverse_iterator it = reader_level_vector.rbegin();
	for (; it != reader_level_vector.rend(); it++, --depth)
	{
		const std::vector<AbstractReaderModel*>& row_vector = *it;

		bool prev_has_child = true;

		NP_2DSHAPE rc_prev;
		rc_prev.pt.x = start_x;
		rc_prev.pt.y = depth*m_grid_layout.grid_size.height;
		rc.sz = m_grid_layout.grid_size;
		for (neuro_u32 i = 0; i < row_vector.size(); i++)
		{
			AbstractReaderModel* model = row_vector[i];

			model_layout_vector.resize(model_layout_vector.size() + 1);
			_MODEL_LAYOUT& layout = model_layout_vector.back();

			layout.right_layout = NULL;
			if (model_layout_vector.size() > 1)
				model_layout_vector[model_layout_vector.size() - 2].right_layout = &layout;

			layout.model = model;
			layout.grid_rc = rc_prev;

			_model_child_map::iterator it_child = model_child_map.find(model);
			if (it_child == model_child_map.end())	// 자식이 없으면 왼쪽 바로 다음에 위치하도록 한다.
			{
				layout.grid_rc.pt.x = rc_prev.Right();

				prev_has_child = false;
			}
			else// child가 있으면 child 중 중간에 위치하도록 한다.
			{
				std::vector<_MODEL_LAYOUT*>& child_vector = it_child->second;
				_MODEL_LAYOUT* middel = child_vector[child_vector.size() / 2];
				layout.grid_rc.pt.x = middel->grid_rc.pt.x - m_grid_layout.grid_size.width / 2;

				// 만약 위치가 이전 layout 다음에 위치하지 않으면
				// 이 layout 뿐만 아니라 첫번째 자손들을 포함 오른쪽의 위치를 죄다 조정해 줘야 한다.
				if (layout.grid_rc.pt.x < rc_prev.Right())
				{
					long move = rc_prev.Right() - layout.grid_rc.pt.x;

					layout.grid_rc.pt.x = rc_prev.Right();

					std::function<void(_MODEL_LAYOUT* first_child)> move_child;
					move_child = [&](_MODEL_LAYOUT* first_child) -> void
					{
						_MODEL_LAYOUT* child = first_child;
						do
						{
							child->grid_rc.pt.x += move;	// 같은 level의 오른쪽을 모두 옮긴다.
							last_x = min(last_x, child->grid_rc.Right());

							child = child->right_layout;
						} while (child);

						_model_child_map::iterator it_child_child = model_child_map.find(first_child->model);
						if (it_child_child != model_child_map.end())
							move_child(it_child_child->second[0]);
					};
					move_child(child_vector[0]);
				}

				prev_has_child = true;
			}

			rc_prev = layout.grid_rc;

			last_x = min(last_x, layout.grid_rc.Right());

			if (layout.model->GetInput())
			{
				std::vector<_MODEL_LAYOUT*>& child_vector = model_child_map[layout.model->GetInput()];
				child_vector.push_back(&layout);
			}
		}
	}
}
*/

