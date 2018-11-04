#pragma once

#include "AbstractNNWnd.h"

#include "project/BindingViewManager.h"

#include "NeuroKernel/network/OutputLayer.h"

using namespace network;
using namespace project;

class SimDisplaySetupNetworkWnd : public NetworkViewManager, public AbstractNNWnd
{
public:
	SimDisplaySetupNetworkWnd(const network::NetworkMatrix& network_matrix, AbstractBindingViewManager& binding_view);
	virtual ~SimDisplaySetupNetworkWnd();

public:
	virtual void ResetSelect() override;
	virtual void RefreshView() override;

	void SelectNetworkLayer(network::AbstractLayer* layer) override;

	bool GetDataBoundLinePoints(const NP_POINT& from_point, const NeuroBindingModel& model, bool& is_hide, gui::_bezier_pt_vector& points) const override;
	inline MATRIX_POINT GetLayerLocation(const AbstractLayer& layer) const override
	{
		return m_network_matrix.GetLayerMatrixPoint(layer);
	}

protected:
	virtual void OnScrollChanged() override;

	void Draw(CDC& dc, CRect rcClient) override;

	void OnLClickedUnit(bool bMouseDown, NP_POINT point) override;
	void OnRClickedUnit(bool bMouseDown, NP_POINT point) override;
};
