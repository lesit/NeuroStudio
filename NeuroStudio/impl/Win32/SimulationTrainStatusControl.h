#pragma once

#include "SimulationStatusControl.h"

#include "util/LogFileWriter.h"

struct _ANALYSIS_EPOCH_INFO
{
	const engine::analysis_loss::_epoch_info* epoch;
	const engine::analysis_loss::_epoch_info* test_epoch;
};

class SimulationTrainStatusControl : public SimulationStatusControl
{
public:
	SimulationTrainStatusControl(SimulationRunningWnd& displayWnd, const _SIM_TRAIN_SETUP_INFO* setup_info);
	virtual ~SimulationTrainStatusControl();

	void ResetProperties();
	void InitStatusListCtrl(CMFCPropertyGridCtrl& status_list_ctrl) override;

	simulate::Simulator* CreateSimulatorInstance(dp::preprocessor::InitShareObject& init_share_object
		, dp::model::ProviderModelManager& ipd
		, engine::NeuralNetworkEngine& network);

	engine::_sigout NetworkSignalProcess(_NET_SIGNAL_INFO& info) override;
	void NetworkBatchSignalProcess(const BATCH_STATUS_INFO& info) override;

protected:
	void learn_start_signal(const _LEARN_START_INFO& info);
	void learn_end_signal(const _LEARN_END_INFO& info);

	void epoch_start_signal(const _EPOCH_START_INFO& info);
	engine::_sigout epoch_end_signal(const _EPOCH_END_INFO& info);

	void SetOnehotResultList(const _output_analysis_onehot_vector& output_analysis_onehot_vector, const bool has_test_after_lean);

private:
	const _SIM_TRAIN_SETUP_INFO* m_setup_info;

	CMFCPropertyGridProperty* m_epoch_group_prop;
	CMFCPropertyGridProperty* m_batch_prop;
	CMFCPropertyGridProperty* m_epoch_prop;
	CMFCPropertyGridProperty* m_epoch_elapse_prop;
	CMFCPropertyGridProperty* m_batch_lr_prop;
	CMFCPropertyGridProperty* m_batch_loss_prop;
	CMFCPropertyGridProperty* m_loss_prop;
	CMFCPropertyGridProperty* m_accord_prop;
	CMFCPropertyGridProperty* m_accuracy_prop;
	CMFCPropertyGridProperty* m_read_elapse_prop;
	CMFCPropertyGridProperty* m_forward_elapse_prop;
	CMFCPropertyGridProperty* m_backward_elapse_prop;

	CMFCPropertyGridProperty* m_test_after_learn_prop;
	CMFCPropertyGridProperty* m_test_after_learn_batch_prop;
	CMFCPropertyGridProperty* m_test_loss_prop;
	CMFCPropertyGridProperty* m_test_accord_prop;
	CMFCPropertyGridProperty* m_test_accuracy_prop;
	CMFCPropertyGridProperty* m_cur_batch_prop;

private:
	util::LogFileWriter m_logFileWriter;
};
