#include "stdafx.h"
#include "BindingViewManager.h"

using namespace np::project;


AbstractBindedViewManager::AbstractBindedViewManager(AbstractBindingViewManager& binding_view)
	: m_binding_view(binding_view)
{

}

project::NeuroStudioProject* AbstractBindedViewManager::GetProject()
{
	return m_binding_view.GetProject();
}

const project::NeuroStudioProject* AbstractBindedViewManager::GetProject() const
{
	return m_binding_view.GetProject();
}

project::NeuroSystemManager* AbstractBindedViewManager::GetNSManager()
{
	if (m_binding_view.GetProject())
		return &m_binding_view.GetProject()->GetNSManager();
	return NULL;
}

const project::NeuroSystemManager* AbstractBindedViewManager::GetNSManager() const
{
	if (m_binding_view.GetProject())
		return &m_binding_view.GetProject()->GetNSManager();
	return NULL;
}

AbstractBindingViewManager::AbstractBindingViewManager(const std::vector<DataViewManager*>& source_vector, NetworkViewManager& network_view)
	: m_network_view(network_view)
{
	m_source_vector = source_vector;

	m_selected_link = NULL;
	m_mouse_over_link = NULL;

	m_is_dragged = false;
	m_drag_start_point.Set(-1, -1);
}

void AbstractBindingViewManager::LoadView()
{
	for (neuro_u32 i = 0; i < m_source_vector.size(); i++)
		m_source_vector[i]->LoadView();

	m_network_view.LoadView();

	MakeBindingLineVector();
}

void AbstractBindingViewManager::SaveView()
{
	for (neuro_u32 i = 0; i < m_source_vector.size(); i++)
		m_source_vector[i]->SaveView();

	m_network_view.SaveView();
}

void AbstractBindingViewManager::ClearBindingLineVector()
{
	m_binding_link_vector.clear();
	m_selected_link = NULL;
	m_mouse_over_link = NULL;
}

void AbstractBindingViewManager::MakeBindingLineVector()
{
	_NEURO_BINDING_LINK* new_select_link = NULL;

	_binding_source_vector source_model_vector;
	for (neuro_u32 i = 0; i < m_source_vector.size(); i++)
		m_source_vector[i]->GetBindedModelVector(source_model_vector);

	const _line_arrow_type arrow_type= GetLineArrowType();

	m_binding_link_vector.resize(source_model_vector.size());
	for (neuro_u32 i = 0; i < source_model_vector.size(); i++)
	{
		_NEURO_BINDING_LINK& link = m_binding_link_vector[i];
		link.line.draw_type = _line_draw_type::bezier;
		link.line.arrow_type = arrow_type;
		link.line.dash_type = _line_dash_type::dot;

		const _BINDING_SOURCE_MODEL& source = source_model_vector[i];

		link.line.bezier.start = source.from_point;
		link.line.bezier.points.clear();

		bool is_hide = false;
		if(!m_network_view.GetDataBoundLinePoints(source.from_point, *source.to, is_hide, link.line.bezier.points))
			continue;

		if (is_hide)
			link.line.arrow_type = _line_arrow_type::none;

		link.from = source.from;
		link.to = source.to;

		if (m_selected_link)
		{
			if (source.from == m_selected_link->from && source.to == m_selected_link->to)
			{
				new_select_link = &link;
				m_selected_link = NULL;
			}
		}
	}
	m_selected_link = new_select_link;

	RefreshBindingViews();
}

void AbstractBindingViewManager::SetBindingMouseoverLink(const _NEURO_BINDING_LINK* link)
{
	if (m_mouse_over_link == link)
		return;

	m_mouse_over_link = link;
	RefreshBindingViews();
}

void AbstractBindingViewManager::SetBindingSelectLink(const _NEURO_BINDING_LINK* link)
{
	if (m_selected_link == link)
		return;

	if(link)
		InitSelection(NULL);

	m_selected_link = link;
	RefreshBindingViews();
}

void AbstractBindingViewManager::InitSelection(AbstractBindedViewManager* exclude)
{
	m_is_dragged = false;
	m_drag_start_point.Set(-1, -1);
	for (neuro_u32 i = 0; i < m_source_vector.size(); i++)
	{
		AbstractBindedViewManager* view = m_source_vector[i];
		if (exclude != view)
		{
			view->ResetSelect();
			view->RefreshView();
		}
	}
	if (exclude != &m_network_view)
	{
		m_network_view.ResetSelect();
		m_network_view.RefreshView();
	}

	RefreshBindingViews();
}

void AbstractBindingViewManager::RefreshBindingViews()
{
	for (neuro_u32 i = 0; i < m_source_vector.size(); i++)
	{
		AbstractBindedViewManager* view = m_source_vector[i];
		view->RefreshView();
	}
	m_network_view.RefreshView();
}

void AbstractBindingViewManager::RefreshNetworkView()
{
	m_network_view.RefreshView();
}
