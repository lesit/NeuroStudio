#pragma once

#include "afxwin.h"

#include "project/BindingViewManager.h"
#include "network/NNMatrixModify.h"

using namespace np::project;

class CModelGridProperty : public CMFCPropertyGridProperty
{
public:
	CModelGridProperty(const CString& strGroupName, DWORD_PTR dwData=0);
	CModelGridProperty(const CString& strName, const COleVariant& value, LPCTSTR lpszDescr = NULL, DWORD_PTR dwData=0);
	virtual ~CModelGridProperty() {}

	void RemoveAllSubItems();
	
	neuro_u32 index;
};

namespace property
{
	enum class _model_property_type { data_reader, data_producer, neural_network, network_layer };
	class ModelPropertyConfigure
	{
	public:
		ModelPropertyConfigure(CMFCPropertyGridCtrl& list_ctrl)
			: m_list_ctrl(list_ctrl) {}

		virtual ~ModelPropertyConfigure() {}

		virtual _model_property_type GetPropertyType() const = 0;
		virtual std::wstring GetPropertyName() const = 0;

		virtual neuro_u32 GetModelSubType() const { return 0; }
		virtual std::vector<neuro_u32> GetSubTypeVector() const { return{}; }
		virtual const wchar_t* GetSubTypeString(neuro_u32 type) const { return L""; }
		virtual bool SubTypeChange(AbstractBindedViewManager* view, neuro_u32 type) { return false; }

		virtual void CompositeProperties() = 0;
		virtual void PropertyChanged(CModelGridProperty* prop, bool& reload) const = 0;

	protected:
		CMFCPropertyGridCtrl& m_list_ctrl;
	};
}

class ModelPropertyWnd : public CDockablePane
{
public:
	ModelPropertyWnd();
	virtual ~ModelPropertyWnd();

	void Clear();
	void SetModelProperty(DataViewManager& view, AbstractPreprocessorModel* preprocessor);
	void SetModelProperty(NetworkViewManager& view, LastSetLayerEntryVector& last_set_entries, AbstractLayer* layer);
	void SetModelProperty(NetworkViewManager& view, network::NeuralNetwork* network);

	const property::ModelPropertyConfigure* GetConfigure() const { return m_configure; }
protected:
	void LoadConfigure();

	property::ModelPropertyConfigure* m_configure;

protected:
	DECLARE_MESSAGE_MAP()
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnSettingChange(UINT uFlags, LPCTSTR lpszSection);
	afx_msg void OnSetFocus(CWnd* pOldWnd);
	afx_msg LRESULT OnPropertyChanged(WPARAM wParam, LPARAM lParam);
	afx_msg void OnSelchangeTypeCombo();

private:
	neuro_u32 m_combo_height;
	CComboBox m_ctrModelTypeCombo;
	CMFCPropertyGridCtrl m_ctrPropertyGrid;

	CFont m_font;

	AbstractBindedViewManager* m_current_view;
};
