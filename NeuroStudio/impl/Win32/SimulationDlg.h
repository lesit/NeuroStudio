#pragma once

#include "SimLearnSetupDlg.h"
#include "SimPredictSetupDlg.h"

#include "SimDisplaySetupWnd.h"
#include "SimulationRunningWnd.h"
#include "afxcmn.h"

// CSimulationDlg ��ȭ �����Դϴ�.


class CSimulationDlg : public CDialog
{
	DECLARE_DYNAMIC(CSimulationDlg)

public:
	CSimulationDlg(np::project::NeuroStudioProject& project, const network::NetworkMatrix& network_matrix, CWnd* pParent = NULL);   // ǥ�� �������Դϴ�.
	virtual ~CSimulationDlg();

// ��ȭ ���� �������Դϴ�.
	enum { IDD = IDD_SIMULATION };

	simulate::_sim_type GetSimType() const { return m_sim_type; }

	np::project::NeuroStudioProject& GetProject() { return m_project; }
	const np::project::NeuroStudioProject& GetProject() const { return m_project; }

	neuro_u32 GetDisplayPeriodBatch() const { return GetCurrentSimSetupDlg().GetDisplayPeriodBatch(); }

	const _layer_display_setup_matrix_vector& GetDisplayInfo() const{
		return m_simDisplaySetupWnd.GetMatrixDisplayVector();
	}

	_SIM_SETUP_INFO* CreateSetupInfo() const {
		return GetCurrentSimSetupDlg().CreateSetupInfo();
	}

	void ReadySimulation();
	void EndSimulation();

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV �����Դϴ�.
	virtual BOOL OnInitDialog();

	DECLARE_MESSAGE_MAP()
	afx_msg void OnBnClickedRadioTrain();
	afx_msg void OnBnClickedRadioRun();
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnGetMinMaxInfo(MINMAXINFO* lpMMI);
	afx_msg void OnTcnSelchangeTabSimDisplay(NMHDR *pNMHDR, LRESULT *pResult);

	afx_msg void OnClose();

private:
	np::project::NeuroStudioProject& m_project;

	CBrush m_backBrush;

	simulate::_sim_type m_sim_type;

	CSimLearnSetupDlg m_learnSetupDlg;
	CSimPredictSetupDlg m_predictSetupDlg;

	// �Ʒ� �� window�� tab���� ��ȯ��ų �� �ֵ��� ����!
	CTabCtrl m_ctrTab;
	SimDisplaySetupWnd m_simDisplaySetupWnd;
	SimulationRunningWnd m_simRunningWnd;

private:
	void ChangedSimType();

	CAbstractSimSetupDlg& GetCurrentSimSetupDlg();
	const CAbstractSimSetupDlg& GetCurrentSimSetupDlg()const {
		return const_cast<CSimulationDlg*>(this)->GetCurrentSimSetupDlg();
	}
};
