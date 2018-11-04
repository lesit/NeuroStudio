
// TestVS2008View.cpp : implementation of the CNeuroStudioView class
//

#include "stdafx.h"
#include "NeuroStudioApp.h"

#include "NeuroStudioDoc.h"
#include "NeuroStudioView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CNeuroStudioView

IMPLEMENT_DYNCREATE(CNeuroStudioView, CView)

// CNeuroStudioView construction/destruction

CNeuroStudioView::CNeuroStudioView()
{
	// TODO: add construction code here
	m_networkView = NULL;
//	m_systemView = NULL;
}

CNeuroStudioView::~CNeuroStudioView()
{
}
// CNeuroStudioView drawing

void CNeuroStudioView::OnDraw(CDC* /*pDC*/)
{
	CDocument* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;

	// TODO: add draw code for native data here
}

#define IDC_TAB				100
#define IDC_SIMULATION_BTN	101
BEGIN_MESSAGE_MAP(CNeuroStudioView, CView)
	ON_WM_CREATE()
	ON_WM_SIZE()
	ON_NOTIFY(TCN_SELCHANGING, IDC_TAB, OnSelchangingTab)
	ON_NOTIFY(TCN_SELCHANGE, IDC_TAB, OnSelchangedTab)
	ON_BN_CLICKED(IDC_SIMULATION_BTN, OnBnClickedSimulation)
END_MESSAGE_MAP()

// CNeuroStudioView message handlers
int CNeuroStudioView::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if(CView::OnCreate(lpCreateStruct) == -1)
		return -1;

//	ModifyStyle(0, WS_CLIPCHILDREN);

	CRect rectDummy;
	rectDummy.SetRectEmpty();

	if (!m_ctrTab.Create(WS_CHILD | WS_VISIBLE | WS_CLIPCHILDREN, rectDummy, this, IDC_TAB))
	{
		TRACE0("Failed to create tab window\n");
		return -1;      // fail to create
	}

	m_networkView = (DeeplearningDesignView*)AddView(RUNTIME_CLASS(DeeplearningDesignView), L"Neural Network design");
	m_ctrTab.SetCurSel(0);
	OnSelchangedTab(NULL, NULL);

	m_ctrSimulationBtn.Create(L"Simulation", WS_CHILD | WS_VISIBLE | WS_CLIPCHILDREN | WS_TABSTOP | BS_DEFPUSHBUTTON, CRect(), this, IDC_SIMULATION_BTN);
	return 0;
}

CView* CNeuroStudioView::AddView(CRuntimeClass* pViewClass, const CString& strViewLabel)
{
	ASSERT_VALID(this);
	ENSURE(pViewClass != NULL);
	ENSURE(pViewClass->IsDerivedFrom(RUNTIME_CLASS(CView)));

	CView* pView = DYNAMIC_DOWNCAST(CView, pViewClass->CreateObject());
	ASSERT_VALID(pView);

	if (!pView->Create(NULL, _T(""), WS_CHILD | WS_CLIPCHILDREN, CRect(0, 0, 0, 0), &m_ctrTab, (UINT)-1, NULL))
	{
		TRACE1("CTabView:Failed to create view '%s'\n", pViewClass->m_lpszClassName);
		return NULL;
	}

	CDocument* pDoc = GetDocument();
	if (pDoc != NULL)
	{
		ASSERT_VALID(pDoc);

		BOOL bFound = FALSE;
		for (POSITION pos = pDoc->GetFirstViewPosition(); !bFound && pos != NULL;)
		{
			if (pDoc->GetNextView(pos) == pView)
			{
				bFound = TRUE;
			}
		}

		if (!bFound)
		{
			pDoc->AddView(pView);
		}
	}

	m_ctrTab.InsertItem(m_ctrTab.GetItemCount(), strViewLabel);
	return pView;
}

void CNeuroStudioView::OnInitialUpdate()
{
	__super::OnInitialUpdate();

}

void CNeuroStudioView::OnSize(UINT nType, int cx, int cy)
{
	CView::OnSize(nType, cx, cy);

	// Tab control should cover a whole client area:
	CRect rc;
	GetClientRect(rc);
	m_ctrTab.MoveWindow(rc);

	CRect rcSimBtn;
	rcSimBtn.right = cx - 5;
	rcSimBtn.left = rcSimBtn.right - 80;
	rcSimBtn.top = rcSimBtn.top + 1;
	rcSimBtn.bottom = rcSimBtn.top + 22;
	m_ctrSimulationBtn.MoveWindow(rcSimBtn);

	m_ctrTab.AdjustRect(FALSE, rc);

	if (!m_networkView)
		return;
//	if (!m_systemView)
//		return;

	m_networkView->MoveWindow(rc);
//	m_systemView->MoveWindow(rc);
}

void CNeuroStudioView::OnSelchangingTab(NMHDR* pNMHDR, LRESULT* pResult)
{
	CView* pActiveView = NULL;
	switch (m_ctrTab.GetCurSel())
	{
	case 0:
		pActiveView = m_networkView;
		break;
	case 1:
//		pActiveView = m_systemView;
		break;
	}
	if (pActiveView)
		pActiveView->ShowWindow(SW_HIDE);
}

void CNeuroStudioView::OnSelchangedTab(NMHDR* pNMHDR, LRESULT* pResult)
{
	CView* pActiveView = NULL;

	_view_vector activate_vector;
	_view_vector deactivate_vector;
	switch (m_ctrTab.GetCurSel())
	{
	case 0:
		pActiveView = m_networkView;
		activate_vector.push_back(&m_networkView->GetProviderWnd());
		activate_vector.push_back(&m_networkView->GetNetworkWnd());
		break;
	case 1:
		deactivate_vector.push_back(&m_networkView->GetProviderWnd());
		deactivate_vector.push_back(&m_networkView->GetNetworkWnd());

//		pActiveView = m_systemView;
//		pViewManager = m_systemView;
		break;
	}
	if (pActiveView)
	{
		pActiveView->BringWindowToTop();
		pActiveView->ShowWindow(SW_SHOWNORMAL);
	}

	for (neuro_u32 i = 0; i < activate_vector.size(); i++)
		activate_vector[i]->Activated();
	for (neuro_u32 i = 0; i < deactivate_vector.size(); i++)
		deactivate_vector[i]->Deactivated();
}

#include "MainFrm.h"
#include "SimulationDlg.h"
void CNeuroStudioView::OnBnClickedSimulation()
{
	np::project::NeuroStudioProject* project = m_networkView->GetProject();
	if (!project)
		return;

	network_ready_error::ReadyError* error = project->ReadyValidationCheck();
	if (error)
	{
		CMainFrame* frame = (CMainFrame*)AfxGetMainWnd();
		DesignErrorOutputPane& error_output_wnd = frame->GetDesignErrorOutputPane();
		error_output_wnd.SetDesignView(m_networkView);
		error_output_wnd.SetErrorList(error);

		MessageBox(error->GetString(), L"Simulation");

		delete error;
		return;
	}

	bool bCancel;
	if (!project->SaveNetworkStructure(false, true, &bCancel))
	{
		MessageBox(L"failed to save network", L"Simulation");
		DEBUG_OUTPUT(L"SaveNetworkStructure : failed");
		return;
	}
	DEBUG_OUTPUT(L"SaveNetworkStructure : completed");

	CSimulationDlg dlg(*project, m_networkView->GetNetworkWnd().GetNetworkMatrix());
	dlg.DoModal();
}
