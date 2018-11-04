#pragma once

#include "AbstractSimSetupDlg.h"
#include "SimulationRunningWnd.h"
#include "gui/win32/PaintWnd.h"

// CSimPredictSetupDlg 대화 상자입니다.

class CSimPredictSetupDlg : public CAbstractSimSetupDlg
{
public:
	CSimPredictSetupDlg(np::project::NeuroStudioProject& project, SimulationRunningWnd& run_wnd);
	virtual ~CSimPredictSetupDlg();

	_SIM_SETUP_INFO* CreateSetupInfo() const override;

	void SaveConfig() override;

// 대화 상자 데이터입니다.
	enum { IDD = IDD_SIM_RUN_SETUP };

protected:
	UINT GetBottomChildWindowID() const override;
	void GetAutoMovingChildArray(CUIntArray& idArray) const override;
	void GetAutoSizingChildArray(CUIntArray& idArray) const override;

	gui::win32::PaintWnd m_ctrPaint;
	CEdit m_ctrEdit;

	const dp::model::DataProviderModel& m_provider_model;

	int m_nType;

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.
	virtual BOOL OnInitDialog();

	void InitDataTree();

	DECLARE_MESSAGE_MAP()
	afx_msg void OnChangedRadio();
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg LRESULT OnEndDrawing(WPARAM wParam, LPARAM lParam);

private:
	CString m_strOutputFilePath;
	CString m_strOutputNoPrefix;

	neuro_u32 m_minibatch_size;
};
