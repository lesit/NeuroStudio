// SimulationTrainDlg.cpp : 구현 파일입니다.
//

#include "stdafx.h"
#include "SimulationDlg.h"
#include "afxdialogex.h"
#include "util/FileUtil.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CSimulationTrainDlg 대화 상자입니다.

IMPLEMENT_DYNAMIC(CSimulationDlg, CDialog)

CSimulationDlg::CSimulationDlg(np::project::NeuroStudioProject& project, const network::NetworkMatrix& network_matrix, CWnd* pParent /*=NULL*/)
: CDialog(CSimulationDlg::IDD, pParent)
, m_learnSetupDlg(project, m_simRunningWnd)
, m_predictSetupDlg(project, m_simRunningWnd)
, m_simDisplaySetupWnd(project, network_matrix)
, m_simRunningWnd(*this)
, m_project(project)
{
	m_backBrush.CreateSolidBrush(RGB(255, 255, 255));

	m_sim_type = simulate::_sim_type::train;
}

CSimulationDlg::~CSimulationDlg()
{
}

void CSimulationDlg::DoDataExchange(CDataExchange* pDX)
{
	__super::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_TAB_SIM_DISPLAY, m_ctrTab);
}

BEGIN_MESSAGE_MAP(CSimulationDlg, CDialog)
	ON_BN_CLICKED(IDC_RADIO_TRAIN, OnBnClickedRadioTrain)
	ON_BN_CLICKED(IDC_RADIO_RUN, OnBnClickedRadioRun)
	ON_WM_SIZE()
	ON_WM_GETMINMAXINFO()
	ON_WM_CLOSE()
	ON_NOTIFY(TCN_SELCHANGE, IDC_TAB_SIM_DISPLAY, &CSimulationDlg::OnTcnSelchangeTabSimDisplay)
END_MESSAGE_MAP()

BOOL CSimulationDlg::OnInitDialog()
{
	DEBUG_OUTPUT(L"start");

	__super::OnInitDialog();

	CButton* pTrainBtn = (CButton*)GetDlgItem(IDC_RADIO_TRAIN);
	if (!pTrainBtn)
		return FALSE;

	pTrainBtn->SetCheck(BST_CHECKED);

	m_learnSetupDlg.Create(m_learnSetupDlg.IDD, this);
	m_predictSetupDlg.Create(m_predictSetupDlg.IDD, this);

	m_ctrTab.InsertItem(0, L"Display setup");
	m_ctrTab.InsertItem(1, L"Simulation");

	DWORD dwDefaultStyle = WS_CHILD;// | WS_CLIPCHILDREN;// | WS_CLIPSIBLINGS;
	CRect rcDummy(0, 0, 0, 0);
	if (!m_simDisplaySetupWnd.Create(NULL, NULL, dwDefaultStyle | WS_VISIBLE | WS_BORDER, rcDummy, this, IDC_SIM_SETUP_WND))
		return FALSE;

	if (!m_simRunningWnd.Create(NULL, NULL, dwDefaultStyle | WS_VISIBLE, rcDummy, this, IDC_SIM_DISPLAY_WND))
		return FALSE;
/*
	if (!m_simRunningWnd.Create(m_simRunningWnd.IDD, this))
		return FALSE;
		*/
	m_simRunningWnd.ShowWindow(SW_HIDE);

	ChangedSimType();

	DEBUG_OUTPUT(L"end");

	SendMessage(WM_SIZE);

	return TRUE;  // return TRUE unless you set the focus to a control
	// 예외: OCX 속성 페이지는 FALSE를 반환해야 합니다.
}

void CSimulationDlg::OnSize(UINT nType, int cx, int cy)
{
	__super::OnSize(nType, cx, cy);

	if (m_learnSetupDlg.GetSafeHwnd() == NULL)
		return;

	CRect rcClient;
	GetClientRect(rcClient);

	CRect rcSetup;
	m_learnSetupDlg.GetWindowRect(rcSetup);
	long width = rcSetup.Width();
	rcSetup.left = rcClient.left + 5;
	rcSetup.right = rcSetup.left + width;
	rcSetup.top = 50;
	rcSetup.bottom = rcClient.bottom - 5;
	m_learnSetupDlg.MoveWindow(rcSetup);
	m_predictSetupDlg.MoveWindow(rcSetup);

	CRect rc;
	rc.top = 0;
	rc.left = rcSetup.right + 15;
	rc.right = rcClient.right - 5;
	rc.bottom = rcSetup.bottom;
	m_ctrTab.MoveWindow(rc);

	m_ctrTab.AdjustRect(FALSE, rc);
	m_simDisplaySetupWnd.MoveWindow(rc);
	m_simRunningWnd.MoveWindow(rc);
}

void CSimulationDlg::OnGetMinMaxInfo(MINMAXINFO* lpMMI)
{
	CRect rc(0, 0, 1500, 700);

	AdjustWindowRect(rc, GetStyle(), FALSE);
	lpMMI->ptMinTrackSize.x = rc.Width();
	lpMMI->ptMinTrackSize.y = rc.Height();

	__super::OnGetMinMaxInfo(lpMMI);
}

void CSimulationDlg::OnBnClickedRadioTrain()
{
	if (m_sim_type == simulate::_sim_type::train)
		return;

	m_sim_type = simulate::_sim_type::train;

	ChangedSimType();
}

void CSimulationDlg::OnBnClickedRadioRun()
{
	if (m_sim_type == simulate::_sim_type::predict)
		return;

	m_sim_type = simulate::_sim_type::predict;

	ChangedSimType();
}

void CSimulationDlg::ChangedSimType()
{
	m_learnSetupDlg.ShowWindow(m_sim_type == simulate::_sim_type::train ? SW_SHOW : SW_HIDE);
	m_predictSetupDlg.ShowWindow(m_sim_type != simulate::_sim_type::train ? SW_SHOW : SW_HIDE);

	m_simRunningWnd.SimTypeChanged();
	SendMessage(WM_SIZE);
}

CAbstractSimSetupDlg& CSimulationDlg::GetCurrentSimSetupDlg()
{
	if (m_sim_type == simulate::_sim_type::train)
		return m_learnSetupDlg;
	else
		return m_predictSetupDlg;
}

void CSimulationDlg::OnTcnSelchangeTabSimDisplay(NMHDR *pNMHDR, LRESULT *pResult)
{
	m_project.SaveProject();

	if (m_ctrTab.GetCurSel() == 0)
	{
		m_simDisplaySetupWnd.ShowWindow(SW_SHOWNORMAL);
		m_simDisplaySetupWnd.SetActiveWindow();
		m_simRunningWnd.ShowWindow(SW_HIDE);
	}
	else
	{
		m_simDisplaySetupWnd.ShowWindow(SW_HIDE);
		m_simRunningWnd.ShowWindow(SW_SHOWNORMAL);
		m_simRunningWnd.SetActiveWindow();
		m_simRunningWnd.DisplaySetupChanged();
	}

	*pResult = 0;
}

void CSimulationDlg::ReadySimulation()
{
	UINT disable_ctrl_array[] = { IDC_RADIO_TRAIN, IDC_RADIO_RUN};
	for (int i = 0; i < _countof(disable_ctrl_array); i++)
		GetDlgItem(disable_ctrl_array[i])->EnableWindow(FALSE);

	GetCurrentSimSetupDlg().BeforeRun();
}

void CSimulationDlg::EndSimulation()
{
	UINT enable_ctrl_array[] = { IDC_RADIO_TRAIN, IDC_RADIO_RUN };
	for (int i = 0; i < _countof(enable_ctrl_array); i++)
		GetDlgItem(enable_ctrl_array[i])->EnableWindow(TRUE);

	GetCurrentSimSetupDlg().AfterRun();
}

void CSimulationDlg::OnClose()
{
	if (!m_simRunningWnd.IsCompletedInitNetwork())
	{
		CString strStatus;
		strStatus.Format(L"You can not close because in initialization.. Please wait and try again.");
		MessageBox(strStatus, L"Simulation");
		return;
	}
	if (m_simRunningWnd.IsRunning())
	{
		CString strStatus;
		strStatus.Format(L"You can not close because in running.. Please click stop button.");
		MessageBox(strStatus, L"Simulation");
		return;
	}

	m_learnSetupDlg.SaveConfig();
	m_predictSetupDlg.SaveConfig();

	m_project.SaveProject();

	__super::OnClose();
}

