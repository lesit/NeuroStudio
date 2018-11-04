#pragma once

#include "AbstractSimSetupDlg.h"

class CSimLearnSetupDlg : public CAbstractSimSetupDlg
{
public:
	CSimLearnSetupDlg(np::project::NeuroStudioProject& project, SimulationRunningWnd& run_wnd);

	virtual ~CSimLearnSetupDlg();

	enum { IDD = IDD_SIM_TRAIN_SETUP };

public:
	void ViewAnalysisHistory();

	_SIM_SETUP_INFO* CreateSetupInfo() const override;

	void BeforeRun() override;
	void AfterRun() override;

	void SaveConfig() override;

protected:
	UINT GetBottomChildWindowID() const override;
	void GetAutoMovingChildArray(CUIntArray& idArray) const override;
	void GetAutoSizingChildArray(CUIntArray& idArray) const override;

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.
	virtual BOOL OnInitDialog();

	DECLARE_MESSAGE_MAP()
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnBnClickedRadioTrain();
	afx_msg void OnBnClickedRadioTest();

private:
	void SetWndCtrlStatus();

	bool m_is_test;

	neuro_u32 m_minibatch_size;

	neuro_u64 m_max_epoch;
	BOOL m_is_stop_below_error;
	neuron_error m_below_error;

	BOOL m_bTestAfterLearn;
	BOOL m_bAnalyzeArgmaxAccuracy;
	BOOL m_bAnalyzeLossHistory;

protected:
	void InitDataTree();
};
