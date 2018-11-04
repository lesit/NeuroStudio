#include "stdafx.h"

#include "AbstractSimSetupDlg.h"
#include "SimulationDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

using namespace np::gui::win32;

CAbstractSimSetupDlg::CAbstractSimSetupDlg(np::project::NeuroStudioProject& project, SimulationRunningWnd& run_wnd)
	: m_project(project), m_run_wnd(run_wnd)
{
	m_backBrush.CreateSolidBrush(RGB(255, 255, 255));
}

CAbstractSimSetupDlg::~CAbstractSimSetupDlg()
{
}

void CAbstractSimSetupDlg::DoDataExchange(CDataExchange* pDX)
{
	__super::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_EDIT_DISPLAY_PERIOD, m_display_period_batch);
	DDV_MinMaxUInt(pDX, m_display_period_batch, 0, neuro_last32);
}

BEGIN_MESSAGE_MAP(CAbstractSimSetupDlg, CDialog)
	ON_WM_SIZE()
	ON_WM_CTLCOLOR()
	ON_EN_CHANGE(IDC_EDIT_DISPLAY_PERIOD, OnEnChangeEditDisplayPeriod)
END_MESSAGE_MAP()

void CAbstractSimSetupDlg::OnEnChangeEditDisplayPeriod()
{
	m_display_period_batch = GetDlgItemInt(IDC_EDIT_DISPLAY_PERIOD);
	m_run_wnd.SetDisplayPeriod(m_display_period_batch);
}

void CAbstractSimSetupDlg::OnSize(UINT nType, int cx, int cy)
{
	__super::OnSize(nType, cx, cy);

	CWnd* pBottom = GetDlgItem(GetBottomChildWindowID());
	if (!pBottom)
		return;

	CRect rc;
	pBottom->GetWindowRect(rc);
	ScreenToClient(rc);

	CRect rcClient;
	GetClientRect(rcClient);

	int gap = rcClient.bottom - 8 - rc.bottom;	// cy-5 를 기준으로 한다.

	CUIntArray movingArray;
	GetAutoMovingChildArray(movingArray);
	movingArray.Add(IDC_STATIC_DISPLAY_PERIOD);
	movingArray.Add(IDC_EDIT_DISPLAY_PERIOD);

	CUIntArray sizingArray;
	GetAutoSizingChildArray(sizingArray);
	if (movingArray.GetCount() == 0 && sizingArray.GetCount()==0)
		return;

	for (INT_PTR i = 0; i < movingArray.GetCount(); i++)
	{
		CWnd* pChild = GetDlgItem(movingArray[i]);
		pChild->GetWindowRect(rc);
		ScreenToClient(rc);
		rc.MoveToY(rc.top + gap);

		pChild->MoveWindow(rc);
		pChild->Invalidate();	// because the child control is not redrawed under the group control
	}

	for (INT_PTR i = 0; i < sizingArray.GetCount(); i++)
	{
		CWnd* pChild = GetDlgItem(sizingArray[i]);
		pChild->GetWindowRect(rc);
		ScreenToClient(rc);
		rc.bottom+=gap;

		pChild->MoveWindow(rc);
		pChild->Invalidate();	// because the child control is not redrawed under the group control
	}
	Invalidate();
}

HBRUSH CAbstractSimSetupDlg::OnCtlColor(CDC* pDC, CWnd* pWnd, UINT nCtlColor)
{
	HBRUSH hbr = __super::OnCtlColor(pDC, pWnd, nCtlColor);

//	if (nCtlColor == CTLCOLOR_STATIC)
//		return m_backBrush;

	return hbr;
}

BOOL CAbstractSimSetupDlg::OnInitDialog()
{
	__super::OnInitDialog();

	m_ctrDataTreeWnd.Create(NULL, NULL, WS_VISIBLE | WS_CHILD, CRect(), this, IDC_TREE_DATA);

	return TRUE;  // return TRUE unless you set the focus to a control
	// 예외: OCX 속성 페이지는 FALSE를 반환해야 합니다.
}
