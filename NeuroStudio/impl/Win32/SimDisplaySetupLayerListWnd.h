#pragma once

#include "project/BindingViewManager.h"
#include "gui/win32/MappingWnd.h"
#include "gui/grid.h"

#include "project/SimDefinition.h"

#include "network/NetworkMatrix.h"

#include "SimDisplayLayerMatrixInfo.h"

#include <map>

using namespace project;

// SimDisplaySetupLayerMatrixWnd 로 바꾸자!!
class SimDisplaySetupLayerListWnd : public DataViewManager, public gui::win32::CMappingWnd
{
public:
	SimDisplaySetupLayerListWnd(project::NeuroStudioProject& project, const network::NetworkMatrix& network_matrix, AbstractBindingViewManager& binding_view);
	virtual ~SimDisplaySetupLayerListWnd() {}

	virtual void ResetSelect() override;
	virtual void RefreshView() override
	{
		RefreshDisplay();
	}

	void GetBindedModelVector(_binding_source_vector& model_vector) const override;
	void LoadView() override;
	void SaveView() override;

	void ToggleDisplayType(const MATRIX_POINT& mp, const AbstractLayer& layer);
	void DeleteSelectedDisplay();
	void ClearAllDisplay();

	const _layer_display_setup_matrix_vector& GetMatrixDisplayVector() const { return m_matrix_display_vector; }
protected:
	const network::NeuralNetwork& m_network;
	const network::NetworkMatrix& m_network_matrix;

	_GRID_LAYOUT m_grid_layout;

	NP_SIZE GetScrollTotalViewSize() const override;
	neuro_u32 GetScrollMoving(bool is_horz) const override;

	_layer_display_info_map& m_layer_display_info_map;
	_layer_display_setup_matrix_vector m_matrix_display_vector;

	struct _LAYOUT_INFO
	{
		neuro_u32 row, col;
		LayerDisplaySetup* layout;
	};

	neuro_u32 FindLayoutRow(const MATRIX_POINT& mp, const AbstractLayer& layer) const;

	bool InsertMatrixDisplayInfo(const network::AbstractLayer& layer, const _LAYER_DISPLAY_INFO& display_info, _LAYOUT_INFO& inserted);
	void DeleteMatrixDisplayInfo(neuro_u32 row, neuro_u32 col);

protected:
	virtual void OnScrollChanged() override;

	virtual void Draw(CDC& dc, CRect rcClient) override;
	virtual void MouseLClickEvent(bool bMouseDown, NP_POINT pt) override;
	virtual void MouseRClickEvent(bool bMouseDown, NP_POINT pt) override;
	virtual void MouseMoveEvent(NP_POINT point) override;

	virtual void ContextMenuEvent(NP_POINT point) override;
	virtual void ProcessMenuCommand(studio::_menu menuID) override;


protected:
	bool m_bMouseLButtonDown;

	_LAYOUT_INFO m_selectedLayout;
	LayerDisplaySetup* m_mouseoverLayout;

	NP_POINT m_insert_point;

protected:
	_LAYOUT_INFO LayoutHitTest(const NP_POINT& point) const;

	void SelectNeuroUnit(NP_POINT point);

	void ShowConfigProperty();
};
