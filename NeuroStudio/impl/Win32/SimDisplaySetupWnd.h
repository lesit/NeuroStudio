#pragma once

#include "project/BindingViewManager.h"

#include "project/NeuroStudioProject.h"

#include "SimDisplaySetupLayerListWnd.h"
#include "SimDisplaySetupNetworkWnd.h"

#include "TitleWnd.h"

class SimDisplaySetupWnd : public CWnd, public AbstractBindingViewManager
{
public:
	SimDisplaySetupWnd(project::NeuroStudioProject& project, const network::NetworkMatrix& network_matrix);
	virtual ~SimDisplaySetupWnd();

	project::NeuroStudioProject* GetProject() override { return &m_project; }

	void ToggleDisplayType(const MATRIX_POINT& mp, const AbstractLayer& layer)
	{
		m_layerSetupListWnd.ToggleDisplayType(mp, layer);
	}

	void ShowConfigProperty(LayerDisplaySetup* layout);

	const _layer_display_setup_matrix_vector& GetMatrixDisplayVector() const {
		return m_layerSetupListWnd.GetMatrixDisplayVector();
	}

protected:
	DECLARE_MESSAGE_MAP()
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg HBRUSH OnCtlColor(CDC* pDC, CWnd* pWnd, UINT nCtlColor);
	afx_msg LRESULT OnPropertyChanged(WPARAM wParam, LPARAM lParam);

private:
	CTitleWnd m_listTitle;
	SimDisplaySetupLayerListWnd m_layerSetupListWnd;
	CTitleWnd m_networkTitle;
	SimDisplaySetupNetworkWnd m_networkWnd;

	CBrush m_backBrush;

	CStatic m_ctrPropertyStatic;
	CMFCPropertyGridCtrl m_layerDisplayProperty;

	LayerDisplaySetup* m_cur_layout;

private:
	project::NeuroStudioProject& m_project;
public:
	afx_msg void OnDestroy();
};
