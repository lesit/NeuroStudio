#include "stdafx.h"

#include "ModelPropertyWnd.h"

using namespace property;

CModelGridProperty::CModelGridProperty(const CString& strGroupName, DWORD_PTR dwData)
	: CMFCPropertyGridProperty(strGroupName, dwData)
{

}
CModelGridProperty::CModelGridProperty(const CString& strName, const COleVariant& value, LPCTSTR lpszDescr, DWORD_PTR dwData)
	: CMFCPropertyGridProperty(strName, value, lpszDescr, dwData)
{
	index = 0;
}

void CModelGridProperty::RemoveAllSubItems()
{
	if (!m_pWndList)
		return;

	CMFCPropertyGridProperty* cur_sel = m_pWndList->GetCurSel();

	for (POSITION pos = m_lstSubItems.GetHeadPosition(); pos != NULL;)
	{
		POSITION posSaved = pos;

		CMFCPropertyGridProperty* pListProp = m_lstSubItems.GetNext(pos);
		ASSERT_VALID(pListProp);

		if (m_pWndList != NULL && cur_sel == pListProp)
			m_pWndList->SetCurSel(NULL, FALSE);

		delete pListProp;
	}
	m_lstSubItems.RemoveAll();
}

ModelPropertyWnd::ModelPropertyWnd()
{
	m_combo_height = 0;
	m_configure = NULL;
	m_current_view = NULL;
}

ModelPropertyWnd::~ModelPropertyWnd()
{
	if (m_configure)
		delete m_configure;
}

#define IDC_MODEL_TYPE_COMBO 100
BEGIN_MESSAGE_MAP(ModelPropertyWnd, CDockablePane)
	ON_WM_CREATE()
	ON_CBN_SELCHANGE(IDC_MODEL_TYPE_COMBO, OnSelchangeTypeCombo)
	ON_REGISTERED_MESSAGE(AFX_WM_PROPERTY_CHANGED, OnPropertyChanged)
	ON_WM_SIZE()
	ON_WM_SETTINGCHANGE()
	ON_WM_SETFOCUS()
END_MESSAGE_MAP()

int ModelPropertyWnd::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (__super::OnCreate(lpCreateStruct) == -1)
		return -1;

	m_ctrModelTypeCombo.Create(WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST | WS_BORDER | WS_CLIPSIBLINGS | WS_CLIPCHILDREN
		, CRect(), this, IDC_MODEL_TYPE_COMBO);

	CRect rectCombo;
	m_ctrModelTypeCombo.GetClientRect(&rectCombo);
	m_combo_height = rectCombo.Height();

	if (!m_ctrPropertyGrid.Create(WS_VISIBLE | WS_CHILD | WS_BORDER, CRect(), this, IDC_PROP_LIST))
	{
		TRACE0("Failed to create Properties Grid \n");
		return -1;      // fail to create
	}
	m_ctrPropertyGrid.EnableHeaderCtrl(FALSE);
	m_ctrPropertyGrid.EnableDescriptionArea();
	m_ctrPropertyGrid.SetVSDotNetLook();
	m_ctrPropertyGrid.MarkModifiedProperties();
	
	return 0;
}

void ModelPropertyWnd::OnSize(UINT nType, int cx, int cy)
{
	__super::OnSize(nType, cx, cy);

	if (GetSafeHwnd() == NULL || (AfxGetMainWnd() != NULL && AfxGetMainWnd()->IsIconic()))
		return;

	if (m_configure)
	{
		m_ctrModelTypeCombo.MoveWindow(0, 0, cx, m_combo_height);
		m_ctrPropertyGrid.MoveWindow(0, m_combo_height, cx, cy - (m_combo_height));
	}
	else
		m_ctrPropertyGrid.MoveWindow(0, 0, cx, cy);
}

void ModelPropertyWnd::OnSettingChange(UINT uFlags, LPCTSTR lpszSection)
{
	__super::OnSettingChange(uFlags, lpszSection);

	::DeleteObject(m_font.Detach());

	LOGFONT lf;
	afxGlobalData.fontRegular.GetLogFont(&lf);

	NONCLIENTMETRICS info;
	info.cbSize = sizeof(info);

	afxGlobalData.GetNonClientMetrics(info);

	lf.lfHeight = info.lfMenuFont.lfHeight;
	lf.lfWeight = info.lfMenuFont.lfWeight;
	lf.lfItalic = info.lfMenuFont.lfItalic;

	m_font.CreateFontIndirect(&lf);

	m_ctrModelTypeCombo.SetFont(&m_font);
	m_ctrPropertyGrid.SetFont(&m_font);
}


void ModelPropertyWnd::OnSetFocus(CWnd* pOldWnd)
{
	__super::OnSetFocus(pOldWnd);

	m_ctrPropertyGrid.SetFocus();
}

void ModelPropertyWnd::Clear()
{
	delete m_configure;
	m_configure = NULL;
	m_current_view = NULL;

	m_ctrPropertyGrid.RemoveAll();
	m_ctrPropertyGrid.RedrawWindow();

	m_ctrModelTypeCombo.ResetContent();
	m_ctrModelTypeCombo.ShowWindow(SW_HIDE);

	CRect rc;
	GetClientRect(&rc);
	SendMessage(WM_SIZE, 0, MAKELPARAM(rc.Width(), rc.Height()));
}

#include "property_configure/DataReaderPropertyConfigure.h"
#include "property_configure/DataProducerPropertyConfigure.h"
void ModelPropertyWnd::SetModelProperty(DataViewManager& view, AbstractPreprocessorModel* preprocessor)
{
	if (preprocessor->GetModelType() == _model_type::reader)
	{
		if (m_configure == NULL || m_configure->GetPropertyType() != _model_property_type::data_reader)
		{
			delete m_configure;
			m_configure = new DataReaderPropertyConfigure(m_ctrPropertyGrid, (AbstractReaderModel*)preprocessor);
		}
		else
			((DataReaderPropertyConfigure*)m_configure)->ChangeModel((AbstractReaderModel*)preprocessor);
	}
	else if (preprocessor->GetModelType() == _model_type::producer)
	{
		if (m_configure == NULL || m_configure->GetPropertyType() != _model_property_type::data_producer)
		{
			delete m_configure;
			m_configure = new DataProducerPropertyConfigure(m_ctrPropertyGrid, (AbstractProducerModel*)preprocessor);
		}
		else
			((DataProducerPropertyConfigure*)m_configure)->ChangeModel((AbstractProducerModel*)preprocessor);
	}
	else
	{
		Clear();
		return;
	}

	m_current_view = &view;
	LoadConfigure();
}

#include "property_configure/LayerPropertyConfigure.h"
void ModelPropertyWnd::SetModelProperty(NetworkViewManager& view, LastSetLayerEntryVector& last_set_entries, AbstractLayer* layer)
{
	if (m_configure == NULL || m_configure->GetPropertyType() != _model_property_type::network_layer)
	{
		delete m_configure;
		m_configure = new LayerPropertyConfigure(m_ctrPropertyGrid, view, last_set_entries, layer);
	}
	else
		((LayerPropertyConfigure*)m_configure)->ChangeLayer(layer);

	m_current_view = &view;
	LoadConfigure();
}

#include "property_configure/NetworkPropertyConfigure.h"
void ModelPropertyWnd::SetModelProperty(NetworkViewManager& view, network::NeuralNetwork* network)
{
	if (m_configure == NULL || m_configure->GetPropertyType() != _model_property_type::neural_network)
	{
		delete m_configure;
		m_configure = new NetworkPropertyConfigure(m_ctrPropertyGrid, network);
	}
	else
		((NetworkPropertyConfigure*)m_configure)->ChangeNetwork(network);

	m_current_view = &view;
	LoadConfigure();
}

void ModelPropertyWnd::LoadConfigure()
{
	m_ctrModelTypeCombo.ResetContent();
	m_ctrPropertyGrid.RemoveAll();

	if (m_configure == NULL)
		return;

	m_ctrModelTypeCombo.ShowWindow(SW_SHOWNORMAL);

	const std::vector<neuro_u32> type_vector = m_configure->GetSubTypeVector();
	m_ctrModelTypeCombo.EnableWindow(type_vector.size() > 1);
	if (type_vector.size() > 0)
	{
		for(int i=0;i<type_vector.size();i++)
		{
			CString model_name = m_configure->GetSubTypeString(type_vector[i]);
			int insert = m_ctrModelTypeCombo.AddString(model_name);
			m_ctrModelTypeCombo.SetItemData(insert, type_vector[i]);

			if(type_vector[i] == m_configure->GetModelSubType())
				m_ctrModelTypeCombo.SetCurSel(i);
		}
	}
	else
	{
		CString str = m_configure->GetPropertyName().c_str();
		m_ctrModelTypeCombo.AddString(str);
		m_ctrModelTypeCombo.SetCurSel(0);
	}
	CRect rc;
	GetClientRect(&rc);
	SendMessage(WM_SIZE, 0, MAKELPARAM(rc.Width(), rc.Height()));

	m_configure->CompositeProperties();
	m_ctrPropertyGrid.RedrawWindow();

	ShowPane(TRUE, FALSE, TRUE);
}

// model type이 바뀌었을때. 즉, BinaryReaderModel -> TextReaderModel 또는 NumericProducer->NlpProducer 등등
void ModelPropertyWnd::OnSelchangeTypeCombo()
{
	neuro_u32 changed_type = m_ctrModelTypeCombo.GetItemData(m_ctrModelTypeCombo.GetCurSel());
	if (changed_type != m_configure->GetModelSubType())
	{
		if (m_configure->SubTypeChange(m_current_view, changed_type))
			LoadConfigure();
	}
}

LRESULT ModelPropertyWnd::OnPropertyChanged(WPARAM wParam, LPARAM lParam)
{
	CModelGridProperty *pProp = (CModelGridProperty*)lParam;
	if (!pProp) return 1;

	bool reload = false;
	m_configure->PropertyChanged(pProp, reload);
	if (reload)
	{
		m_ctrPropertyGrid.RemoveAll();
		m_configure->CompositeProperties();

		DWORD prev_data = pProp->GetData();
		if (prev_data)
		{
			CMFCPropertyGridProperty* sel = m_ctrPropertyGrid.FindItemByData(prev_data);
			m_ctrPropertyGrid.SetCurSel(sel, FALSE);
			m_ctrPropertyGrid.RedrawWindow();
		}
	}

	if (m_current_view)	// 속성 변경에 따라 producer에 연결된 layer 또는 현재 속성의 layer가 변경될 수 있으므로
		m_current_view->GetBindingView().RefreshNetworkView();
	return 0;
}
