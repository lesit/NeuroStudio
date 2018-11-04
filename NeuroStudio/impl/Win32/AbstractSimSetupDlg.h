#pragma once

#include "project/NeuroStudioProject.h"

#include "SimDataTreeWnd.h"
#include "SimulationTrainStatusControl.h"

#include "SimulationRunningWnd.h"

using namespace np::engine;

class CSimulationDlg;
class CAbstractSimSetupDlg : public CDialog
{
public:
	CAbstractSimSetupDlg(np::project::NeuroStudioProject& project, SimulationRunningWnd& run_wnd);
	virtual ~CAbstractSimSetupDlg();

	neuro_u32 GetDisplayPeriodBatch() const { return m_display_period_batch; }

	virtual _SIM_SETUP_INFO* CreateSetupInfo() const = 0;

	virtual void BeforeRun() {}
	virtual void AfterRun() {}

	virtual void SaveConfig(){};

protected:
	virtual UINT GetBottomChildWindowID() const = 0;
	virtual void GetAutoMovingChildArray(CUIntArray& idArray) const {};
	virtual void GetAutoSizingChildArray(CUIntArray& idArray) const {};

protected:
	void DoDataExchange(CDataExchange* pDX) override;
	virtual BOOL OnInitDialog();

	DECLARE_MESSAGE_MAP()
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg HBRUSH OnCtlColor(CDC* pDC, CWnd* pWnd, UINT nCtlColor);
	afx_msg void OnEnChangeEditDisplayPeriod();
	CBrush m_backBrush;

protected:
	np::project::NeuroStudioProject& m_project;
	SimulationRunningWnd& m_run_wnd;

	neuro_u32 m_display_period_batch;

	CSimDataTreeWnd m_ctrDataTreeWnd;
};
