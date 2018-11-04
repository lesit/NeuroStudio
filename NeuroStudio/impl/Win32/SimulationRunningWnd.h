#pragma once

#include "SimulationDisplayLayerWnd.h"

#include "gui/win32/GroupListCtrl.h"

#include "AnalysisWnd.h"

#include "SimulationTrainStatusControl.h"
#include "SimulationPredictStatusControl.h"

#include "thread/Lock.h"

class CSimulationDlg;
class SimulationRunningWnd : public CWnd, protected dp::JobSignalReciever
	, public engine::RecvSignal
{
public:
	SimulationRunningWnd(CSimulationDlg& simDlg);
	virtual ~SimulationRunningWnd();

	enum { IDD = IDD_SIM_RUNNING};

	void SimTypeChanged();

	bool IsCompletedInitNetwork();
	bool IsRunning() { return m_hRunThread != NULL; }

	void StartRunning();
	void PauseRunning();

	void SetDisplayPeriod(neuro_u32 batch_no) { m_sim_instance.display_period_batch = batch_no; }

	CAnalysisWnd& GetAnalysisWnd() { return m_analysisWnd; }
	CGroupListCtrl& GetOnehotResultListCtrl() { return m_ctrOnehotResultList; }

	void PostSimulationMessage(const wchar_t* str);

	void DisplaySetupChanged();

	// for SimulationStatusControl
	void PatchResults();
	void ClearResult();

protected:
	//	virtual BOOL OnInitDialog();
	DECLARE_MESSAGE_MAP()
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg HBRUSH OnCtlColor(CDC* pDC, CWnd* pWnd, UINT nCtlColor);
	afx_msg void OnPaint();
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnBnClickedRadioParallelType();
	afx_msg void OnBnClickedStart();
	afx_msg void OnBnClickedEnd();
	afx_msg void OnBnClickedButtonStopEpoch();
	afx_msg LRESULT OnInitializedMessage(WPARAM wParam, LPARAM lParam);
	afx_msg LRESULT OnSimNetSignal(WPARAM wParam, LPARAM lParam);
	afx_msg LRESULT OnSimMessage(WPARAM wParam, LPARAM lParam);
	afx_msg LRESULT OnRedrawResults(WPARAM wParam, LPARAM lParam);
	afx_msg LRESULT OnCompletedRunMessage(WPARAM wParam, LPARAM lParam);

	core::math_device_type m_pdType;

	CStatic m_ctrPdTypeStatic;
	CButton m_ctrGpuRadioBtn;
	CButton m_ctrCpuRadioBtn;

	CButton m_ctrStartBtn;
	CButton m_ctrStopEpochBtn;
	CButton m_ctrStopBtn;

	CStatic m_ctrElapseLabelStatic;
	CStatic m_ctrElapseStatic;

	CStatic m_ctrRunStatusStatic;

	CMFCPropertyGridCtrl m_ctrStatusGrid;

	bool m_has_analysis;
	CAnalysisWnd m_analysisWnd;

	SimulationDisplayLayerWnd m_layerDisplayWnd;

	bool m_has_onehot_result;
	CStatic m_ctrOnehotResultStatic;
	CGroupListCtrl m_ctrOnehotResultList;

	CBrush m_backBrush;

protected:
	CSimulationDlg& m_simDlg;

	static DWORD WINAPI IntNetworkThread(LPVOID pParam);
	HANDLE m_hIntNetworkThread;
	thread::Lock m_initThreadLock;
	CString m_strInitFailedMsg;

	dp::preprocessor::InitShareObject m_init_share_object;

	void dataio_job_status(neuro_u32 job_id, const char* status) override;

	enum class _fail_status { none, create_simulator, net_ready, net_run };
	static DWORD WINAPI RunningThread(LPVOID pParam);
	HANDLE m_hRunThread;
	HANDLE m_hAwakeFromPause;

	engine::_PARALLEL_INSTANCE m_p_instance;
	engine::NeuralNetworkEngine* m_network;
	
	bool CreateNetworkInstance(core::math_device_type pd_type);

	_sigout network_signal(const _NET_SIGNAL_INFO& info, std::unordered_set<neuro_u32>* epoch_start_onehot_output = NULL) override;

	enum class _sim_control_type { run, pause, stop, stop_epoch, finish_condition};
	struct _SIMULATION_INSTANCE
	{
		_SIMULATION_INSTANCE()
		{
			sim_control = NULL;
			simulator = NULL;

			sim_control_type = _sim_control_type::run;
			display_period_batch = 0;
			last_elapse = 0.f;
		}
		void Clear()
		{
			simulator = NULL;

			thread::Lock::Owner owner(m_signal_lock);

			delete sim_control;
			sim_control = NULL;
			sim_control_type = _sim_control_type::run;
			display_period_batch = 0;

			for (neuro_u32 i = 0, n = epoch_signal_vector.size(); i < n; i++)
				delete epoch_signal_vector[i];

			epoch_signal_vector.clear();
			batch_signal_vector.clear();
			last_elapse = 0.f;
		}
		SimulationStatusControl* sim_control;
		simulate::Simulator* simulator;
		_sim_control_type sim_control_type;
		neuro_u32 display_period_batch;

		thread::Lock m_signal_lock;
		std::vector<_NET_SIGNAL_INFO*> epoch_signal_vector;
		std::vector<BATCH_STATUS_INFO> batch_signal_vector;
		neuro_float last_elapse;
	};
	_SIMULATION_INSTANCE m_sim_instance;

	thread::Lock m_layer_display_lock;
	_layer_display_item_matrix_vector m_layer_display_item_matrix_vector;
	std::unordered_set<neuro_u32> m_onehot_result_output_set;

private:
	void ReadySimulation(const _SIM_SETUP_INFO& sim_setup_info);
	void EndSimulation();
};
