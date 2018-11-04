#pragma once

#include "SimulationStatusControl.h"

class SimulationPredictStatusControl : public SimulationStatusControl
{
public:
	SimulationPredictStatusControl(SimulationRunningWnd& displayWnd, const _SIM_PREDICT_SETUP_INFO* setup_info);
	virtual ~SimulationPredictStatusControl();

	void InitStatusListCtrl(CMFCPropertyGridCtrl& status_list_ctrl) override;

	simulate::Simulator* CreateSimulatorInstance(dp::preprocessor::InitShareObject& init_share_object
		, dp::model::ProviderModelManager& ipd
		, engine::NeuralNetworkEngine& network) override;

	void ResetProperties();

	engine::_sigout NetworkSignalProcess(_NET_SIGNAL_INFO& info) override;

protected:
	void predict_start_signal(const _PREDICT_START_INFO& info);
	void predict_end_signal(const _PREDICT_END_INFO& info);

private:
	const _SIM_PREDICT_SETUP_INFO* m_setup_info;

	CMFCPropertyGridProperty* m_data_count_prop;
	CMFCPropertyGridProperty* m_batch_prop;

	CMFCPropertyGridProperty* m_elapse_prop;
	CMFCPropertyGridProperty* m_read_elapse_prop;
	CMFCPropertyGridProperty* m_forward_elapse_prop;
};
