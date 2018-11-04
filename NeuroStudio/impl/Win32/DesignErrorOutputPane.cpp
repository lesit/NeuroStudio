#include "stdafx.h"

#include "DesignErrorOutputPane.h"

#include "DeeplearningDesignView.h"

DesignErrorOutputPane::DesignErrorOutputPane()
{
	m_design_view = NULL;
}

DesignErrorOutputPane::~DesignErrorOutputPane()
{
}

#define IDC_ERROR_LIST WM_USER+1

BEGIN_MESSAGE_MAP(DesignErrorOutputPane, CDockablePane)
	ON_WM_CREATE()
	ON_WM_SIZE()
	ON_LBN_SELCHANGE(IDC_ERROR_LIST, OnLbnSelchangeErrorList)
END_MESSAGE_MAP()


int DesignErrorOutputPane::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (__super::OnCreate(lpCreateStruct) == -1)
		return -1;

	if(!m_ctrErrorListBox.Create(WS_CHILD | WS_VISIBLE | WS_BORDER | WS_HSCROLL | WS_VSCROLL | LBS_NOTIFY | LBS_NOINTEGRALHEIGHT
		, CRect(), this, IDC_ERROR_LIST))
		return -1;

	return 0;
}

void DesignErrorOutputPane::OnSize(UINT nType, int cx, int cy)
{
	__super::OnSize(nType, cx, cy);

	if (GetSafeHwnd() == NULL || (AfxGetMainWnd() != NULL && AfxGetMainWnd()->IsIconic()))
		return;

	m_ctrErrorListBox.MoveWindow(0, 0, cx, cy);
}

void DesignErrorOutputPane::SetDesignView(DeeplearningDesignView* design_view)
{
	m_design_view = design_view;
}

void DesignErrorOutputPane::Clear()
{
	m_ctrErrorListBox.ResetContent();
}

#include "gui/win32/WinUtil.h"

void DesignErrorOutputPane::SetErrorList(project::network_ready_error::ReadyError* error)
{
	m_ctrErrorListBox.ResetContent();

	if (error->GetType() == project::network_ready_error::_error_type::no_network)
	{
		m_ctrErrorListBox.AddString(error->GetString());
	}
	else if (error->GetType() == project::network_ready_error::_error_type::layer_error)
	{
		project::network_ready_error::LayersError* layer_error = (project::network_ready_error::LayersError*)error;
		for (neuro_u32 i = 0; i < layer_error->layer_error_vector.size(); i++)
		{
			const project::network_ready_error::_LAYER_ERROR_INFO& info = layer_error->layer_error_vector[i];
			std::wstring msg = info.msg;

			int inserted=m_ctrErrorListBox.AddString(msg.c_str());
			m_ctrErrorListBox.SetItemData(inserted, (DWORD_PTR)info.layer);
		}
	}
	WinUtil::ResizeListBoxHScroll(m_ctrErrorListBox);
	m_ctrErrorListBox.Invalidate();

	ShowPane(TRUE, FALSE, TRUE);
}

void DesignErrorOutputPane::OnLbnSelchangeErrorList()
{
	AbstractLayer* layer = NULL;
	int cur_sel = m_ctrErrorListBox.GetCurSel();
	if(cur_sel!= LB_ERR  && cur_sel < m_ctrErrorListBox.GetCount())
		layer = (AbstractLayer*)m_ctrErrorListBox.GetItemData(cur_sel);

	m_design_view->GetNetworkWnd().SelectLayer(layer);
}
