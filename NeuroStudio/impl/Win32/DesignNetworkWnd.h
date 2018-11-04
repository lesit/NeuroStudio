#pragma once

#include "AbstractNNWnd.h"
#include "project/LastSetLayerEntryVector.h"

#include "project/BindingViewManager.h"

using namespace network;
using namespace np::project;

class DesignNetworkWnd : public AbstractNNWnd, public project::NetworkViewManager
{
public:
	DesignNetworkWnd(DeepLearningDesignViewManager& binding_view);
	virtual ~DesignNetworkWnd();

	virtual void LoadView() override;
	virtual void SaveView() override;

	void RefreshView() override {
		RefreshDisplay();
	}

	bool GetDataBoundLinePoints(const NP_POINT& from_point, const NeuroBindingModel& model, bool& is_hide, gui::_bezier_pt_vector& points) const override;
	inline MATRIX_POINT GetLayerLocation(const AbstractLayer& layer) const override
	{
		return m_network_matrix.GetLayerMatrixPoint(layer);
	}

	void SelectNetworkLayer(network::AbstractLayer* layer) override;

public:
	virtual _drop_test DropTest(const _DRAG_SOURCE& source, NP_POINT point) override;
	virtual bool Drop(const _DRAG_SOURCE& source, NP_POINT point) override;
	virtual void DragLeave() override;

	// layer type이 바뀌면 입력 연결등의 변화가 생기므로 다시 그려야 한다.
	bool ChangeLayerType(HiddenLayer* layer, _layer_type layer_type, const nsas::_LAYER_STRUCTURE_UNION* entry, _slice_input_vector* org_erased_input_vector=NULL);

protected:
	void AfterNetworkSelected(NP_POINT point) override;
	void AfterNetworkMouseMove(NP_POINT point) override;

	virtual void OnScrollChanged() override;

	void Draw(CDC& dc, CRect rcClient) override;

	void OnLClickedUnit(bool bMouseDown, NP_POINT point) override;
	void OnRClickedUnit(bool bMouseDown, NP_POINT point) override;
	bool OnContextMenu(NP_POINT point, const _POS_INFO_IN_LAYER& pos_info) override;

	void ProcessMenuCommand(studio::_menu menuID) override;

protected:
	void DrawDragLine(CDC& dc);

	void BeginDragLayer(NP_POINT pt, const MATRIX_POINT& matrix_pt) override;

	void ResetSelect() override;
	void SelectMultiLayers(const NP_2DSHAPE& rc) override;

	void ShowNetworkProperty();
	bool AddLayer(bool is_output);

private:
	_POS_INFO_IN_LAYER m_insert_layer_pos;

	struct _DROP_TARGET_INFO
	{
		_drop_test dropType;
		_POS_INFO_IN_LAYER pos;
	};
	_DROP_TARGET_INFO m_cur_drop_target;

	np::project::LastSetLayerEntryVector m_last_set_entries;
private:
	NNMatrixModify m_network_matrix_modify;

	CPen m_insert_bar_pen;
	CBrush m_insert_bar_brush;
};
