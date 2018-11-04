#include "stdafx.h"

#include "SimulationPredictStatusControl.h"

#include "SimulationRunningWnd.h"

SimulationPredictStatusControl::SimulationPredictStatusControl(SimulationRunningWnd& displayWnd, const _SIM_PREDICT_SETUP_INFO* setup_info)
	: SimulationStatusControl(displayWnd)
{
	m_setup_info = setup_info;
	ResetProperties();
}

SimulationPredictStatusControl::~SimulationPredictStatusControl()
{
	delete m_setup_info;
}

void SimulationPredictStatusControl::ResetProperties()
{
	m_data_count_prop = NULL;
	m_batch_prop = NULL;

	m_read_elapse_prop = NULL;
	m_forward_elapse_prop = NULL;
}

void SimulationPredictStatusControl::InitStatusListCtrl(CMFCPropertyGridCtrl& status_list_ctrl)
{
	status_list_ctrl.RemoveAll();
	ResetProperties();

	CMFCPropertyGridProperty* data_group_prop = new CMFCPropertyGridProperty(L"Data");
	status_list_ctrl.AddProperty(data_group_prop, FALSE, FALSE);
	m_data_count_prop = new CMFCPropertyGridProperty(L"Total data count", (variant_t)neuro_size_t(0), L"");
	data_group_prop->AddSubItem(m_data_count_prop);
	m_batch_prop = new CMFCPropertyGridProperty(L"Batch info", (variant_t)L"count[0], size[0]", L"");
	data_group_prop->AddSubItem(m_batch_prop);

	CMFCPropertyGridProperty* elapse_group_prop = new CMFCPropertyGridProperty(L"Elapse");
	m_read_elapse_prop = new CMFCPropertyGridProperty(L"Read elapse", (variant_t)L"0.0 s", L"");
	elapse_group_prop->AddSubItem(m_read_elapse_prop);
	m_forward_elapse_prop = new CMFCPropertyGridProperty(L"Forward elapse", (variant_t)L"0.0 s", L"");
	elapse_group_prop->AddSubItem(m_forward_elapse_prop);
}

#include "util/FileUtil.h"
simulate::Simulator* SimulationPredictStatusControl::CreateSimulatorInstance(dp::preprocessor::InitShareObject& init_share_object
	, dp::model::ProviderModelManager& ipd
	, engine::NeuralNetworkEngine& network)
{
	const dp::model::DataProviderModel& provider_model = ipd.GetFinalProvider(true);

	engine::PREDICT_SETUP setup;
	memset(&setup, 0, sizeof(engine::PREDICT_SETUP));

	dp::preprocessor::DataProvider* provider = new dp::preprocessor::DataProvider(init_share_object, false, false, m_setup_info->minibatch_size);
	if (m_setup_info->predict_data_type == _SIM_PREDICT_SETUP_INFO::_predict_data_type::filepath)
	{
		if (m_setup_info->uid_datanames_map.size() == 0)
		{
			delete provider;
			return NULL;
		}

		provider->SetDataSource(m_setup_info->uid_datanames_map);
		if (!provider->Create(provider_model))
		{
			delete provider;
			return NULL;
		}
	}
	else
	{
		const dp::model::_producer_model_vector& producer_model_vector = provider_model.GetProducerVector();
		if (producer_model_vector.size() != 1)
		{
			delete provider;
			return NULL;
		}

		if (m_setup_info->predict_data_type == _SIM_PREDICT_SETUP_INFO::_predict_data_type::text)
		{
			_MEM_DATA_SOURCE source;
			source.size = m_setup_info->text_data.size();
			if (source.size == 0)
			{
				delete provider;
				return NULL;
			}
			source.data = malloc(source.size);
			memcpy(source.data, m_setup_info->text_data.c_str(), source.size);

			_uid_mem_data_map uid_mem_data_map;
			uid_mem_data_map[producer_model_vector[0]->uid] = source;
			provider->SetDataSource(uid_mem_data_map);
			if (!provider->Create(provider_model))
			{
				delete provider;
				return NULL;
			}
		}
		else
		{
			dp::preprocessor::MemProducer* producer = new dp::preprocessor::MemProducer(m_setup_info->img_data.count);
			memcpy(producer->m_value_buffer.buffer, m_setup_info->img_data.buffer, sizeof(neuro_float) * m_setup_info->img_data.count);

			_producer_model_instance_vector producer_model_instance_vector;
			producer_model_instance_vector.push_back({ producer_model_vector[0], producer });

			if (!provider->CreateDirect(producer_model_instance_vector))
			{
				delete provider;
				return NULL;
			}
		}
	}
	setup.provider = provider;

	// 가짜뉴스찾기. 일단 시간이 없으니 여기에서 하자!
	dp::StreamWriter* result_writer = NULL;
	if (!m_setup_info->strOutputFilePath.empty())
	{
		dp::_STREAM_WRITE_INFO write_info;

		device::FileDeviceFactory file_factory(m_setup_info->strOutputFilePath.c_str());
		write_info.device = file_factory.CreateWriteAdaptor(true, false);

		write_info.value_float_length = 0;
		write_info.value_float_under_length = 7;

		write_info.col_delimiter = "\n";
		write_info.row_delimiter = ",";

		dp::_STREAM_WRITE_ROW_INFO row_info;
		row_info.type = dp::_STREAM_WRITE_ROW_INFO::_source_type::ret_text_source;
		row_info.ref_text_source_index = 0;
		write_info.col_vector.push_back(row_info);

		row_info.type = dp::_STREAM_WRITE_ROW_INFO::_source_type::value;
		row_info.value_onehot = false;
		row_info.value_index = 1;	// 1(가짜일) 확률을 구하는 거니까
		write_info.col_vector.push_back(row_info);

		//		dp::preprocessor::TextReader* ref_text_source = provider->at(0).at(0)->GetTextFilteredReader();
		//		result_writer = new dataio::StreamWriter(write_info, ref_text_source);
	}
	setup.recv_signal = &m_displayWnd;

	return new simulate::PredictSimulator(network, m_setup_info->minibatch_size, setup);
}

engine::_sigout SimulationPredictStatusControl::NetworkSignalProcess(_NET_SIGNAL_INFO& info)
{
	switch (info.GetType())
	{
	case _signal_type::predict_start:
		predict_start_signal((const _PREDICT_START_INFO&)info);
		break;
	case _signal_type::predict_end:
		predict_end_signal((const _PREDICT_END_INFO&)info);
		break;
	}
	return engine::_sigout::sig_continue;
}

void SimulationPredictStatusControl::predict_start_signal(const _PREDICT_START_INFO& info)
{
	m_data_count_prop->SetValue((variant_t)info.data.data_count);
	std::wstring str = util::StringUtil::Format<wchar_t>(L"count[%llu], size[%llu]", info.data.batch_count, info.data.batch_size);
	m_batch_prop->SetValue((variant_t)str.c_str());
}

void SimulationPredictStatusControl::predict_end_signal(const _PREDICT_END_INFO& info)
{
	std::wstring elapse_str;
	SimulationStatusControl::GetElapseString(info.batch_gen_elapse, elapse_str);
	m_read_elapse_prop->SetValue(elapse_str.c_str());

	SimulationStatusControl::GetElapseString(info.forward_elapse, elapse_str);
	m_forward_elapse_prop->SetValue(elapse_str.c_str());
}
