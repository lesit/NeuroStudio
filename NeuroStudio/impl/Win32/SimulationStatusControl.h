#pragma once

#include "simulation/Simulator.h"

#include "NeuroKernel/engine/NeuralNetworkProcessor.h"

using namespace np::engine;

struct _SIM_SETUP_INFO
{
	virtual ~_SIM_SETUP_INFO() {}

	virtual simulate::_sim_type GetRunType() const = 0;

	neuro_u32 minibatch_size;
};

struct _SIM_TRAIN_SETUP_INFO : public _SIM_SETUP_INFO
{
	virtual ~_SIM_TRAIN_SETUP_INFO() {}

	simulate::_sim_type GetRunType() const override { return simulate::_sim_type::train; }

	LEARNNING_SETUP learn_info;

	bool useNdf;
	bool data_noising;

	_uid_datanames_map uid_datanames_map;
	_uid_datanames_map test_uid_datanames_map;

//	neuro_size_t max_epoch;
	bool is_stop_below_error;
	neuron_error below_error;
};

struct _SIM_PREDICT_SETUP_INFO : public _SIM_SETUP_INFO
{
	simulate::_sim_type GetRunType() const override { return simulate::_sim_type::predict; }

	virtual ~_SIM_PREDICT_SETUP_INFO()
	{
		img_data.Dealloc();
	}

	enum class _predict_data_type { filepath, text, image };
	_predict_data_type predict_data_type;

	_uid_datanames_map uid_datanames_map;
	std::string text_data;
	_VALUE_VECTOR img_data;

	std::string strOutputFilePath;
};

union SIM_BATCH_INFO
{
	struct
	{
		neuro_u32 no;
		neuro_float learn_rate;
	};
	neuro_u64 u64_data;
};

class SimulationRunningWnd;
class SimulationStatusControl
{
public:
	SimulationStatusControl(SimulationRunningWnd& displayWnd);
	virtual ~SimulationStatusControl() {}

	virtual void InitStatusListCtrl(CMFCPropertyGridCtrl& status_list_ctrl) = 0;

	virtual simulate::Simulator* CreateSimulatorInstance(dp::preprocessor::InitShareObject& init_share_object
		, dp::model::ProviderModelManager& ipd
		, NeuralNetworkEngine& network) = 0;

	static void GetElapseString(neuro_float elapse, std::wstring& str);

	virtual engine::_sigout NetworkSignalProcess(_NET_SIGNAL_INFO& info) = 0;
	virtual void NetworkBatchSignalProcess(const BATCH_STATUS_INFO& info) {}

protected:
	SimulationRunningWnd& m_displayWnd;
};
