#include "stdafx.h"

#include "SimDisplaySetupWnd.h"

SimDisplaySetupWnd::SimDisplaySetupWnd(project::NeuroStudioProject& project, const network::NetworkMatrix& network_matrix)
	: AbstractBindingViewManager({ &m_layerSetupListWnd }, m_networkWnd)
	, m_layerSetupListWnd(project, network_matrix, *this), m_networkWnd(network_matrix, *this), m_project(project)
{
	m_cur_layout = NULL;
	m_backBrush.CreateSolidBrush(RGB(255, 255, 255));
}

SimDisplaySetupWnd::~SimDisplaySetupWnd()
{
}

#define IDC_SETUP_LIST (WM_USER+1)
#define IDC_NETWORK_WND (WM_USER+2)

BEGIN_MESSAGE_MAP(SimDisplaySetupWnd, CWnd)
	ON_WM_CREATE()
	ON_WM_CTLCOLOR()
	ON_WM_SIZE()
	ON_REGISTERED_MESSAGE(AFX_WM_PROPERTY_CHANGED, OnPropertyChanged)
	ON_WM_DESTROY()
END_MESSAGE_MAP()

int SimDisplaySetupWnd::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (__super::OnCreate(lpCreateStruct) == -1)
		return -1;

	CRect rcDummy(0, 0, 0, 0);

	DWORD dwTitleStyle = WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS;
	DWORD dwStyle = WS_CHILD | WS_VISIBLE | WS_HSCROLL | WS_VSCROLL;

	m_listTitle.Create(L"Display", dwTitleStyle | WS_BORDER, rcDummy, this, 270);
	if (!m_layerSetupListWnd.Create(NULL, NULL, dwStyle, rcDummy, this, IDC_SETUP_LIST))
		return -1;

	m_networkTitle.Create(L"Network", dwTitleStyle, rcDummy, this, 270);
	if (!m_networkWnd.Create(NULL, NULL, dwStyle | WS_BORDER, rcDummy, this, IDC_NETWORK_WND))
		return -1;

	m_ctrPropertyStatic.Create(L"Display property", WS_CHILD | WS_VISIBLE | SS_CENTERIMAGE, rcDummy, this);

	if (!m_layerDisplayProperty.Create(WS_VISIBLE | WS_CHILD | WS_BORDER, rcDummy, this, IDC_PROP_LIST))
		return -1;
	m_layerDisplayProperty.EnableHeaderCtrl(FALSE);
	m_layerDisplayProperty.SetVSDotNetLook();

	LoadView();
	return 0;
}

void SimDisplaySetupWnd::OnDestroy()
{
	SaveView();

	__super::OnDestroy();
}

HBRUSH SimDisplaySetupWnd::OnCtlColor(CDC* pDC, CWnd* pWnd, UINT nCtlColor)
{
	HBRUSH hbr = __super::OnCtlColor(pDC, pWnd, nCtlColor);

	if (nCtlColor == CTLCOLOR_STATIC)
	{
		pDC->SetTextColor(RGB(0, 0, 0));
		pDC->SetBkColor(RGB(255, 255, 255));
		return m_backBrush;
	}

	return hbr;
}

void SimDisplaySetupWnd::OnSize(UINT nType, int cx, int cy)
{
	__super::OnSize(nType, cx, cy);

	if (m_layerSetupListWnd.GetSafeHwnd() == NULL)
		return;

	CRect rcClient;
	GetClientRect(&rcClient);

	const neuro_u32 title_height = 30;

	CRect rcList(0, 0, cx, cy / 3);
	m_listTitle.MoveWindow(0, rcList.top, title_height, rcList.bottom);
	rcList.left = title_height;
	m_layerSetupListWnd.MoveWindow(rcList);

	CRect rc=rcList;
	rc.left = cx - 230;
	rc.right = cx;
	rc.bottom = rc.top + 25;
	m_ctrPropertyStatic.MoveWindow(rc);

	rc.top = rc.bottom;
	rc.bottom = rc.top + 100;
	m_layerDisplayProperty.MoveWindow(rc);

	{
		CRgn rgn;
		rgn.CreateRectRgn(0, 0, rcList.Width(), rcList.Height());

		CRgn prop_rgn; prop_rgn.CreateRectRgn(rc.left - rcList.left, rcList.top, rc.right - rcList.left, rc.bottom);
		rgn.CombineRgn(&rgn, &prop_rgn, RGN_DIFF);

		m_layerSetupListWnd.SetWindowRgn(rgn, TRUE);
	}

	m_networkTitle.MoveWindow(0, rcList.bottom, title_height, cy - rcList.bottom);
	m_networkWnd.MoveWindow(title_height, rcList.bottom, cx - title_height, cy - rcList.bottom);
}

#include "desc/LayerDesc.h"
void SimDisplaySetupWnd::ShowConfigProperty(LayerDisplaySetup* layout)
{
	m_layerDisplayProperty.RemoveAll();

	m_cur_layout = layout;
	if (layout == NULL)
	{
		m_layerDisplayProperty.RedrawWindow();
		return;
	}

	CMFCPropertyGridProperty* prop = new CMFCPropertyGridProperty(L"Layer"
		, (variant_t)str_rc::LayerDesc::GetSimpleName(*layout->layer)
		, str_rc::LayerDesc::GetDetailName(*layout->layer).c_str());
	prop->AllowEdit(FALSE);
	m_layerDisplayProperty.AddProperty(prop);

	prop = new CMFCPropertyGridProperty(L"Location", (variant_t)layout->mp.ToString().c_str(), L"");
	prop->AllowEdit(FALSE);
	m_layerDisplayProperty.AddProperty(prop);

	prop = new CMFCPropertyGridProperty(L"Display type"
		, (variant_t)ToString(layout->display.type), L""
		, (DWORD_PTR)&layout->display.type);
	for (neuro_u32 i = 0; i < _countof(layer_display_type_string); i++)
		prop->AddOption(layer_display_type_string[i]);
	prop->AllowEdit(FALSE);
	m_layerDisplayProperty.AddProperty(prop);

	if (layout->layer->GetLayerType() == network::_layer_type::output)
	{
		const bool is_read_label = ((network::OutputLayer*)layout->layer)->ReadLabelForTarget();
		if(is_read_label)
			layout->display.is_argmax_output = true;

		if (layout->display.type != project::_layer_display_type::none)
		{
			prop = new CMFCPropertyGridProperty(L"Argmax"
				, (variant_t)layout->display.is_argmax_output
				, L"Whether display argmax"
				, (DWORD_PTR)&layout->display.is_argmax_output);

			prop->Enable(!is_read_label);
			m_layerDisplayProperty.AddProperty(prop);
		}
		if (is_read_label)
		{
			prop = new CMFCPropertyGridProperty(L"Onehot results"
				, (variant_t)layout->display.is_onehot_analysis_result
				, L"Anaysis view of onehot results"
				, (DWORD_PTR)&layout->display.is_onehot_analysis_result);
			m_layerDisplayProperty.AddProperty(prop);
		}
	}
}

LRESULT SimDisplaySetupWnd::OnPropertyChanged(WPARAM wParam, LPARAM lParam)
{
	CMFCPropertyGridProperty *pProp = (CMFCPropertyGridProperty*)lParam;
	if (!pProp) return 1;

	DWORD_PTR data = pProp->GetData();
	if (data == (DWORD_PTR)&m_cur_layout->display.type)
	{
		_layer_display_type old_type = m_cur_layout->display.type;

		CString value = pProp->GetValue();
		for (neuro_u32 i = 0; i < _countof(layer_display_type_string); i++)
		{
			if (value == layer_display_type_string[i])
			{
				m_cur_layout->display.type = (_layer_display_type)i;
				break;
			}
		}
		if (old_type!= _layer_display_type::none && m_cur_layout->display.type == _layer_display_type::none
			|| old_type == _layer_display_type::none && m_cur_layout->display.type != _layer_display_type::none)
		{
			m_layerSetupListWnd.ToggleDisplayType(m_cur_layout->mp, *m_cur_layout->layer);
		}
	}
	else if(data == (DWORD_PTR)&m_cur_layout->display.is_argmax_output)
	{
		m_cur_layout->display.is_argmax_output = pProp->GetValue().boolVal;
	}
	else if (data == (DWORD_PTR)&m_cur_layout->display.is_onehot_analysis_result)
	{
		m_cur_layout->display.is_onehot_analysis_result = pProp->GetValue().boolVal;
	}
	return 0;
}
