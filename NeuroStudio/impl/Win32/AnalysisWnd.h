#pragma once

#include "LossGraphWnd.h"

#include "SimulationTrainStatusControl.h"

class CAnalysisWnd : public CWnd
{
	DECLARE_DYNAMIC(CAnalysisWnd)

public:
	CAnalysisWnd();
	virtual ~CAnalysisWnd();

	void ReadySimulation(const _SIM_TRAIN_SETUP_INFO& sim_setup_info);

	void AddHistory(const _ANALYSIS_EPOCH_INFO& epoch_info);

	void Clear();

protected:
	DECLARE_MESSAGE_MAP()
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg HBRUSH OnCtlColor(CDC* pDC, CWnd* pWnd, UINT nCtlColor);
	afx_msg void OnSize(UINT nType, int cx, int cy);

private:
	CBrush m_backBrush;

	CStatic m_ctrEpochHistoryStatic;
	CListCtrl m_ctrEpochHistoryList;

	CStatic m_ctrEpochGraphStatic;
	CLossGraphWnd m_ctrEpochGraphWnd;

	CStatic m_ctrTestEpochGraphStatic;
	CLossGraphWnd m_ctrTestEpochGraphWnd;

protected:
	bool m_has_argmax_accuracy;
	bool m_has_second_history;
};
