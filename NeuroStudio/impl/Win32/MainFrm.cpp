
// MainFrm.cpp : CMainFrame 클래스의 구현
//

#include "stdafx.h"
#include "NeuroStudioApp.h"

#include "MainFrm.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CMainFrame

IMPLEMENT_DYNCREATE(CMainFrame, CFrameWndEx)

BEGIN_MESSAGE_MAP(CMainFrame, CFrameWndEx)
	ON_WM_CREATE()
	ON_COMMAND(ID_VIEW_PROPERTIES_WND, OnViewPropertiesWnd)
	ON_COMMAND(ID_VIEW_ERROR_WND, OnViewErrorWnd)
	ON_WM_GETMINMAXINFO()
END_MESSAGE_MAP()

static UINT indicators[] =
{
	ID_SEPARATOR,           // 상태 줄 표시기
	ID_INDICATOR_CAPS,
	ID_INDICATOR_NUM,
	ID_INDICATOR_SCRL,
};

// CMainFrame 생성/소멸

CMainFrame::CMainFrame()
{
}

CMainFrame::~CMainFrame()
{
}

int CMainFrame::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CFrameWndEx::OnCreate(lpCreateStruct) == -1)
		return -1;

	CMFCVisualManager::SetDefaultManager(RUNTIME_CLASS(CMFCVisualManagerVS2008));
	RedrawWindow(NULL, NULL, RDW_ALLCHILDREN | RDW_INVALIDATE | RDW_UPDATENOW | RDW_FRAME | RDW_ERASE);

	if (!m_wndMenuBar.Create(this))
	{
		TRACE0("메뉴 모음을 만들지 못했습니다.\n");
		return -1;      // 만들지 못했습니다.
	}

	m_wndMenuBar.SetPaneStyle(m_wndMenuBar.GetPaneStyle() | CBRS_SIZE_DYNAMIC | CBRS_TOOLTIPS | CBRS_FLYBY);

	// 메뉴 모음을 활성화해도 포커스가 이동하지 않게 합니다.
	CMFCPopupMenu::SetForceMenuFocus(FALSE);

	if (!m_wndStatusBar.Create(this))
	{
		TRACE0("상태 표시줄을 만들지 못했습니다.\n");
		return -1;      // 만들지 못했습니다.
	}
	m_wndStatusBar.SetIndicators(indicators, sizeof(indicators)/sizeof(UINT));

	// TODO: 도구 모음 및 메뉴 모음을 도킹할 수 없게 하려면 이 다섯 줄을 삭제하십시오.
	m_wndMenuBar.EnableDocking(CBRS_ALIGN_ANY);
	EnableDocking(CBRS_ALIGN_ANY);
	DockPane(&m_wndMenuBar);

	// enable Visual Studio 2005 style docking window behavior
	CDockingManager::SetDockingMode(DT_SMART);
	// enable Visual Studio 2005 style docking window auto-hide behavior
	EnableAutoHidePanes(CBRS_ALIGN_ANY);

	// create docking windows
	if (!CreateDockingWindows())
	{
		TRACE0("Failed to create docking windows\n");
		return -1;
	}

	m_property_pane.EnableDocking(CBRS_ALIGN_ANY);
	DockPane(&m_property_pane);

	m_design_error_pane.EnableDocking(CBRS_ALIGN_ANY);
	CDockablePane* pTabbedBar = NULL;
	m_design_error_pane.AttachToTabWnd(&m_property_pane, DM_SHOW, FALSE, &pTabbedBar);

	// 메뉴 개인 설정을 활성화합니다(가장 최근에 사용한 명령).
	// TODO: 사용자의 기본 명령을 정의하여 각 풀다운 메뉴에 하나 이상의 기본 명령을 포함시킵니다.
	CList<UINT, UINT> lstBasicCommands;

	lstBasicCommands.AddTail(ID_FILE_NEW);
	lstBasicCommands.AddTail(ID_FILE_OPEN);
	lstBasicCommands.AddTail(ID_FILE_SAVE);
	lstBasicCommands.AddTail(ID_FILE_SAVE_AS);
	lstBasicCommands.AddTail(ID_NEURALNETWORK_REPLACE);
	lstBasicCommands.AddTail(ID_APP_EXIT);
	lstBasicCommands.AddTail(ID_APP_ABOUT);
	lstBasicCommands.AddTail(ID_VIEW_PROPERTIES_WND);
	lstBasicCommands.AddTail(ID_VIEW_ERROR_WND);

	CMFCToolBar::SetBasicCommands(lstBasicCommands);

	return 0;
}

bool CMainFrame::CreateDockingWindows()
{
	if (!m_property_pane.Create(L"Properties", this, CRect(0, 0, 300, 200), TRUE, IDC_PROPERTIESWND, WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS | WS_CLIPCHILDREN | CBRS_RIGHT | CBRS_FLOAT_MULTI))
	{
		TRACE0("Failed to create Properties window\n");
		return false; // failed to create
	}

	if (!m_design_error_pane.Create(L"Design Error", this, CRect(0, 0, 300, 200), TRUE, IDC_DESIGNERRORWND, WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS | WS_CLIPCHILDREN | CBRS_RIGHT | CBRS_FLOAT_MULTI))
	{
		TRACE0("Failed to create Properties window\n");
		return false; // failed to create
	}
	return true;
}

void CMainFrame::OnGetMinMaxInfo(MINMAXINFO* lpMMI)
{
	CRect rc(0, 0, 800, 500);

	AdjustWindowRect(rc, GetStyle(), FALSE);
	lpMMI->ptMinTrackSize.x = rc.Width();
	lpMMI->ptMinTrackSize.y = rc.Height();

	__super::OnGetMinMaxInfo(lpMMI);
}

void CMainFrame::OnViewPropertiesWnd()
{
	m_property_pane.ShowPane(TRUE, FALSE, TRUE);
}

void CMainFrame::OnViewErrorWnd()
{
	m_design_error_pane.ShowPane(TRUE, FALSE, TRUE);
}

#ifdef _DEBUG
void CMainFrame::AssertValid() const
{
	CFrameWndEx::AssertValid();
}

void CMainFrame::Dump(CDumpContext& dc) const
{
	CFrameWndEx::Dump(dc);
}
#endif //_DEBUG
