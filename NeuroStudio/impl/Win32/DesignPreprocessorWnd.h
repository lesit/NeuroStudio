#pragma once

#include "gui/win32/MappingWnd.h"
#include "gui/grid.h"
#include "gui/Win32/GraphicUtil.h"

#include "project/BindingViewManager.h"

#include "NeuroData/model/DataProviderModel.h"

#include "DesignNetworkWnd.h"

using namespace np;
using namespace np::gui::win32;
using namespace np::dp;
using namespace np::dp::model;

typedef std::vector<dp::model::AbstractPreprocessorModel*> _data_row_vector;
typedef std::vector<_data_row_vector> _data_level_vector;
class DesignPreprocessorWnd : public CMappingWnd, public project::DataViewManager
{
public:
	DesignPreprocessorWnd(DeepLearningDesignViewManager& binding_view);
	virtual ~DesignPreprocessorWnd();

	virtual void LoadView() override;
	virtual void SaveView() override;
	
	virtual void ResetSelect() override;
	void RefreshView() override {
		RefreshDisplay();
	}

	void GetBindedModelVector(_binding_source_vector& model_vector) const override;

	void SetIntegratedLayout(bool is_integrated);

	virtual _drop_test DropTest(const _DRAG_SOURCE& source, NP_POINT point) override;
	virtual bool Drop(const _DRAG_SOURCE& source, NP_POINT point) override;
	virtual void DragLeave() override;

	NP_SIZE GetScrollTotalViewSize() const override;
	neuro_u32 GetScrollMoving(bool is_horz) const override;

	bool ReplacePreprocessorModel(AbstractPreprocessorModel* old_model, AbstractPreprocessorModel* new_model);

protected:
	void ClearAll();
	void CompositeAll();

	struct _MODEL_LAYOUT
	{
		NP_2DSHAPE rc;
		AbstractPreprocessorModel* model;
	};
	typedef std::vector<_MODEL_LAYOUT> _model_layout_vector;
	typedef std::vector<_model_layout_vector> _layout_level_vector;

	typedef std::vector<_reader_model_vector> _reader_level_vector;
	void CompositeModelVector(const _producer_model_vector& producer_vector, const _reader_level_vector& reader_level_vector
								, long start_x, neuro_u32 level_count
								, _layout_level_vector& layout_level_vector, NP_POINT& max_end_pt);
	void CompositeReaderLevelVector(const _reader_model_vector& reader_vector, _reader_level_vector& reader_level_vector);

private:
	_GRID_LAYOUT m_grid_layout;
	neuro_u32 m_source_desc_height;

	_layout_level_vector m_predict_layout_vector;
	_layout_level_vector m_learn_layout_vector;

	neuro_32 m_seperate_vertical_line_x;
	neuro_u32 m_max_width;
	neuro_u32 m_max_height;

	NP_RECT m_producer_scope_rect;

	struct _LINK_INFO
	{
		~_LINK_INFO() {}

		bool HasLink() const { return from!=NULL && to!=NULL; }
		bool operator==(const _LINK_INFO& src) const { return from == src.from && to == src.to; }
		bool operator!=(const _LINK_INFO& src) const { return from != src.from || to != src.to; }

		AbstractPreprocessorModel *from, *to;

		_CURVE_INTEGRATED_LINE line;
	};
	std::vector<_LINK_INFO> m_link_vector;

protected:
	bool m_bMouseLButtonDown;

	AbstractPreprocessorModel* m_mouseoverModel;
	const _LINK_INFO* m_mouseoverLink;

	struct _SELECTED_UNIT
	{
		_SELECTED_UNIT() { Initialize(); }

		void Initialize() {
			link = NULL;
			model = NULL;
		}
		bool IsValid() const
		{
			return link != NULL || model != NULL;
		}

		const _LINK_INFO* link;
		AbstractPreprocessorModel* model;
	};
	_SELECTED_UNIT m_selected_unit;

	NP_POINT m_insert_point;

	struct _DROP_TARGET_INFO
	{
		_drop_test dropType;
		AbstractPreprocessorModel* model;
	};
	_DROP_TARGET_INFO m_cur_drop_target;

protected:
	virtual void OnScrollChanged() override;

	virtual void Draw(CDC& dc, CRect rcClient) override;
	virtual void MouseLClickEvent(bool bMouseDown, NP_POINT pt) override;
	virtual void MouseRClickEvent(bool bMouseDown, NP_POINT pt) override;
	virtual void MouseMoveEvent(NP_POINT point) override;

	virtual void ContextMenuEvent(NP_POINT point) override;
	virtual void ProcessMenuCommand(studio::_menu menuID) override;

	const _LINK_INFO* LinkHitTest(const NP_POINT& point) const;
	AbstractPreprocessorModel* ModelHitTest(const NP_POINT& point) const;

	void SelectNeuroUnit(NP_POINT point);

	void ShowConfigProperty();

	void BeginDragModel(NP_POINT pt, AbstractPreprocessorModel* model);
};
