#pragma once

#include "gui/win32/MappingWnd.h"
#include "gui/Win32/GraphicUtil.h"

#include "network/NetworkMatrix.h"
#include "network/NNMatrixModify.h"

using namespace np;
using namespace np::network;
using namespace np::gui::win32;

class AbstractNNWnd : public CMappingWnd
{
public:
	AbstractNNWnd(const network::NetworkMatrix& network_matrix, const std::vector<_CLIPBOARDFORMAT_INFO>& cf_vector = {});
	virtual ~AbstractNNWnd();

	void ClearSelect();

	const NetworkMatrix& GetNetworkMatrix() const { return  m_network_matrix; }

	void SelectLayer(AbstractLayer* layer);
	void EnsureVisible(const AbstractLayer& layer);

protected:
	NP_SIZE GetScrollTotalViewSize() const override;
	neuro_u32 GetScrollMoving(bool is_horz) const override;

	virtual void Draw(CDC& dc, CRect rcClient) override;
	virtual void MouseLClickEvent(bool bMouseDown, NP_POINT pt) override;
	virtual void MouseRClickEvent(bool bMouseDown, NP_POINT pt) override;
	virtual void MouseMoveEvent(NP_POINT point) override;
	virtual void ContextMenuEvent(NP_POINT point) override;

protected:
	virtual _POS_INFO_IN_LAYER SelectNeuroUnit(NP_POINT point);
	virtual void AfterNetworkSelected(NP_POINT point) {}
	virtual void AfterNetworkMouseMove(NP_POINT point) {}

	const _LINK_INFO* LinkHitTest(const NP_POINT& pt) const;

	virtual void OnLClickedUnit(bool bMouseDown, NP_POINT point) {}
	virtual void OnRClickedUnit(bool bMouseDown, NP_POINT point) {}

	virtual bool OnContextMenu(NP_POINT point, const _POS_INFO_IN_LAYER& pos_info) { return false; }

protected:

	virtual void BeginDragLayer(NP_POINT pt, const MATRIX_POINT& matrix_pt){}

protected:
	const NetworkMatrix& m_network_matrix;

	bool m_bMouseLButtonDown;

	AbstractLayer* m_mouseoverLayer;
	const _LINK_INFO* m_mouseoverLink;

	struct _SELECTED_UNIT
	{
		_SELECTED_UNIT() { Initialize(); }

		void Initialize() {
			link = NULL;
			layer = NULL;
		}
		bool IsValid() const
		{
			return link != NULL || layer != NULL;
		}

		const _LINK_INFO* link;

		AbstractLayer* layer;
	};
	_SELECTED_UNIT m_selected_unit;
	MATRIX_SCOPE m_selected_scope;

	virtual void SelectMultiLayers(const NP_2DSHAPE& rc);
};
