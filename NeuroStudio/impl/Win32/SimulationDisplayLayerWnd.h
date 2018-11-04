#pragma once

#include "gui/win32/ScrollWnd.h"
#include "gui/Win32/GraphicUtil.h"

#include "simulation/Simulator.h"

#include "SimDisplayLayerMatrixInfo.h"

using namespace network;
using namespace project;
using namespace simulate;
using namespace gui;
using namespace gui::win32;

class SimulationDisplayLayerWnd : public CScrollWnd
{
public:
	SimulationDisplayLayerWnd(const _layer_display_item_matrix_vector& layer_display_item_matrix_vector);
	virtual ~SimulationDisplayLayerWnd();

	void SetSimulatorStatus(bool is_running) {
		m_isSimulatorRunning = is_running;
	}

	void SetDelegateSample(neuro_u32 sample);
	void RedrawResults();

protected:
	DECLARE_MESSAGE_MAP()
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnPaint();

	CFont m_layer_label_font;
	CFont m_argmax_label_font;
	CFont m_list_font;

private:
	struct _LAYER_LAYOUT
	{
		NP_SIZE sz;
		neuro_u32 label_height;
		neuro_u32 argmax_height[2];
	};
	typedef std::vector<_LAYER_LAYOUT> _layout_row_vector;
	typedef std::vector<_layout_row_vector> _layout_matrix;

	NP_SIZE CalcLayoutMatrix(CDC& dc, _layout_matrix& layout_matrix) const;
	void DrawLayerOutput(CDC& dc, const NP_RECT& rcDraw, const LayerDisplayItem& item, const _LAYER_LAYOUT& layout);

protected:
	_GRID_LAYOUT m_max_grid_layout;
	NP_SIZE m_min_item_draw_size;

	static const neuro_u32 m_draw_cell_border;
	static const neuro_u32 m_draw_border;

	bool IsDrawTargetList(const LayerDisplayItem& item) const;

	NP_SIZE GetScrollTotalViewSize() const override;
	neuro_u32 GetScrollMoving(bool is_horz) const override;

	bool m_isSimulatorRunning;

	const _layer_display_item_matrix_vector& m_layer_display_item_matrix_vector;

	neuro_u32 m_delegate_sample;
};
