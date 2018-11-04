#include "StdAfx.h"
#include "SimDisplaySetupNetworkWnd.h"
#include "math.h"

#include "project/NeuroStudioProject.h"
#include "util/StringUtil.h"
#include "gui/Win32/GraphicUtil.h"

#include "SimDisplaySetupWnd.h"

using namespace np::gui;
using namespace np::studio;

// ���߿� ���⼭ ���� dc ���� ��� ������ �÷��� �Ѵ�. ǥ�� c �����Ϸ��� ����ϱ� ����
SimDisplaySetupNetworkWnd::SimDisplaySetupNetworkWnd(const network::NetworkMatrix& network_matrix, AbstractBindingViewManager& binding_view)
	: AbstractNNWnd(network_matrix), NetworkViewManager(binding_view)
{
}

SimDisplaySetupNetworkWnd::~SimDisplaySetupNetworkWnd()
{
}

void SimDisplaySetupNetworkWnd::ResetSelect()
{
	ClearSelect();
}

void SimDisplaySetupNetworkWnd::RefreshView()
{
	RefreshDisplay();
}

bool SimDisplaySetupNetworkWnd::GetDataBoundLinePoints(const NP_POINT& from_point, const NeuroBindingModel& model, bool& is_hide, gui::_bezier_pt_vector& points) const
{
	const AbstractLayer& layer = (const AbstractLayer&)model;
	MATRIX_POINT mp = m_network_matrix.GetLayerMatrixPoint(layer);
	if (m_network_matrix.GetLayer(mp) != &layer)
		return false;

	is_hide = false;

	NP_RECT rc = m_network_matrix.GetLayerRect(mp);
	rc = ViewportToWnd(rc);

	CPoint pt_to(rc.left + (rc.right - rc.left) / 2, rc.top);
	if (pt_to.y <= 0)
	{
		is_hide = true;
		pt_to.y = -2;
	}

	CWnd* parent = GetParent();
	ClientToScreen(&pt_to);
	parent->ScreenToClient(&pt_to);	// �θ���ǥ�� ������

	points.push_back(_BEZIER_POINT({ pt_to.x, pt_to.y }, false, 1.5f));
	return true;
}

void SimDisplaySetupNetworkWnd::SelectNetworkLayer(network::AbstractLayer* layer)
{
	SelectLayer(layer);
}

void SimDisplaySetupNetworkWnd::OnScrollChanged()
{
	m_binding_view.MakeBindingLineVector();
}

void SimDisplaySetupNetworkWnd::Draw(CDC& dc, CRect rcClient)
{
	__super::Draw(dc, rcClient);

	const _binding_link_vector& binding_link_vector = m_binding_view.GetBindingLinkVector();
	DrawBindingLines(dc.GetSafeHdc(), binding_link_vector, m_binding_view.GetBindingSelectedLink(), m_binding_view.GetBindingMouseoverLink());
}

#include "SimDisplaySetupWnd.h"
void SimDisplaySetupNetworkWnd::OnLClickedUnit(bool bMouseDown, NP_POINT point)
{
	if (!bMouseDown)
		return;

	network::_POS_INFO_IN_LAYER pos;
	if(!m_network_matrix.LayerHitTest(WndToViewport(point), pos) || pos.layer == NULL)
		return;
	
	// Ŭ���ϸ� display�� �߰��ϰų� ���� �Ѵ�.
	((SimDisplaySetupWnd&)m_binding_view).ToggleDisplayType(pos.matrix_pt, *pos.layer);
}

void SimDisplaySetupNetworkWnd::OnRClickedUnit(bool bMouseDown, NP_POINT point)
{
	OnLClickedUnit(bMouseDown, point);
}
