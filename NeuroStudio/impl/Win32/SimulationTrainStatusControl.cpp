#include "stdafx.h"

#include "SimulationTrainStatusControl.h"

#include "SimulationRunningWnd.h"
#include "gui/win32/GroupListCtrl.h"
#include "gui/win32/WinUtil.h"

SimulationTrainStatusControl::SimulationTrainStatusControl(SimulationRunningWnd& displayWnd, const _SIM_TRAIN_SETUP_INFO* setup_info)
	: SimulationStatusControl(displayWnd)
{
	ResetProperties();
	
	m_setup_info = setup_info;
}

SimulationTrainStatusControl::~SimulationTrainStatusControl()
{
	delete m_setup_info;
}

void SimulationTrainStatusControl::ResetProperties()
{
	m_epoch_group_prop = NULL;
	m_batch_prop = NULL;
	m_epoch_prop = NULL;
	m_epoch_elapse_prop = NULL;
	m_batch_lr_prop = NULL;
	m_batch_loss_prop = NULL;
	m_loss_prop = NULL;
	m_accord_prop = NULL;
	m_accuracy_prop = NULL;
	m_cur_batch_prop = NULL;
	m_read_elapse_prop = NULL;
	m_forward_elapse_prop = NULL;
	m_backward_elapse_prop = NULL;

	m_test_after_learn_prop = NULL;
	m_test_after_learn_batch_prop = NULL;
	m_test_loss_prop = NULL;
	m_test_accord_prop = NULL;
	m_test_accuracy_prop = NULL;
}

void SimulationTrainStatusControl::InitStatusListCtrl(CMFCPropertyGridCtrl& status_list_ctrl)
{
	status_list_ctrl.RemoveAll();
	ResetProperties();

	CMFCPropertyGridProperty* elapse_group_prop=NULL;
	auto create_last_group_prop = [&]()
	{
		if (m_setup_info->learn_info.learn_type != engine::_learn_type::test)
			m_epoch_group_prop = new CMFCPropertyGridProperty(L"Learn : data[0]");
		else
			m_epoch_group_prop = new CMFCPropertyGridProperty(L"Test : data[0]");
		status_list_ctrl.AddProperty(m_epoch_group_prop, FALSE, FALSE);

		m_batch_prop = new CMFCPropertyGridProperty(L"Batch info", (variant_t)L"count[0], size[0]", L"");
		m_epoch_group_prop->AddSubItem(m_batch_prop);
		m_loss_prop = new CMFCPropertyGridProperty(L"Loss", (variant_t)L"0", L"");
		m_epoch_group_prop->AddSubItem(m_loss_prop);

		if (m_setup_info->learn_info.analyze.bAnalyzeArgmaxAccuracy)
		{
			m_accord_prop = new CMFCPropertyGridProperty(L"Accord", (variant_t)neuro_u32(0), L"");
			m_epoch_group_prop->AddSubItem(m_accord_prop);
			m_accuracy_prop = new CMFCPropertyGridProperty(L"Accuracy", (variant_t)L"0 %", L"");
			m_epoch_group_prop->AddSubItem(m_accuracy_prop);
		}
		m_cur_batch_prop = new CMFCPropertyGridProperty(L"Batch : 0");
		m_epoch_group_prop->AddSubItem(m_cur_batch_prop);

		elapse_group_prop = new CMFCPropertyGridProperty(L"Elapse");
		if(m_setup_info->learn_info.learn_type != engine::_learn_type::test)
			m_epoch_group_prop->AddSubItem(elapse_group_prop);
		else
			status_list_ctrl.AddProperty(elapse_group_prop, FALSE, FALSE);

		m_read_elapse_prop = new CMFCPropertyGridProperty(L"Read elapse", (variant_t)L"0.0 s", L"");
		elapse_group_prop->AddSubItem(m_read_elapse_prop);
		m_forward_elapse_prop = new CMFCPropertyGridProperty(L"Forward elapse", (variant_t)L"0.0 s", L"");
		elapse_group_prop->AddSubItem(m_forward_elapse_prop);
	};

	if (m_setup_info->learn_info.learn_type != engine::_learn_type::test)
	{
		m_epoch_prop = new CMFCPropertyGridProperty(L"Epoch : 0");
		status_list_ctrl.AddProperty(m_epoch_prop, FALSE, FALSE);
		m_epoch_elapse_prop = new CMFCPropertyGridProperty(L"Elapse", (variant_t)L"0.0 s", L"");
		m_epoch_prop->AddSubItem(m_epoch_elapse_prop);

		create_last_group_prop();

		m_batch_lr_prop = new CMFCPropertyGridProperty(L"Learn rate", (variant_t)L"0", L"");
		m_cur_batch_prop->AddSubItem(m_batch_lr_prop);
		m_batch_loss_prop = new CMFCPropertyGridProperty(L"Loss", (variant_t)L"0", L"");
		m_cur_batch_prop->AddSubItem(m_batch_loss_prop);

		// 제일 의미 없으니 맨 마지막에
		m_backward_elapse_prop = new CMFCPropertyGridProperty(L"Learn elapse", (variant_t)L"0.0 s", L"");
		elapse_group_prop->AddSubItem(m_backward_elapse_prop);

		if( m_setup_info->learn_info.learn_type == engine::_learn_type::learn_test_both)
		{
			m_test_after_learn_prop = new CMFCPropertyGridProperty(L"Test : data[0]");
			status_list_ctrl.AddProperty(m_test_after_learn_prop, FALSE, FALSE);
			m_test_after_learn_batch_prop = new CMFCPropertyGridProperty(L"Batch info", (variant_t)L"count[0], size[0]", L"");
			m_test_after_learn_prop->AddSubItem(m_test_after_learn_batch_prop);

			m_test_loss_prop = new CMFCPropertyGridProperty(L"Loss", (variant_t)0.f, L"");
			m_test_after_learn_prop->AddSubItem(m_test_loss_prop);
			if (m_setup_info->learn_info.analyze.bAnalyzeArgmaxAccuracy)
			{
				m_test_accord_prop = new CMFCPropertyGridProperty(L"Accord", (variant_t)neuro_u32(0), L"");
				m_test_after_learn_prop->AddSubItem(m_test_accord_prop);
				m_test_accuracy_prop = new CMFCPropertyGridProperty(L"Accuracy", (variant_t)L"0 %", L"");
				m_test_after_learn_prop->AddSubItem(m_test_accuracy_prop);
			}
		}
	}
	else
	{
		create_last_group_prop();
		m_batch_loss_prop = new CMFCPropertyGridProperty(L"Loss", (variant_t)L"0", L"");
		m_cur_batch_prop->AddSubItem(m_batch_loss_prop);
	}
	status_list_ctrl.AdjustLayout();
	status_list_ctrl.ExpandAll();
	elapse_group_prop->Expand(FALSE);	// 제일 의미 없으니 일단 접어 둔다.

	CGroupListCtrl& ctrOnehotResultList = m_displayWnd.GetOnehotResultListCtrl();
	ctrOnehotResultList.DeleteAllItems();
	gui::win32::WinUtil::DeleteAllColumns(ctrOnehotResultList);

	ctrOnehotResultList.AddColumn(L"class");
	ctrOnehotResultList.AddColumn(L"target");
	ctrOnehotResultList.AddColumn(L"output");
	ctrOnehotResultList.AddColumn(L"accord");
	if (m_setup_info->learn_info.learn_type == engine::_learn_type::learn_test_both)
	{
		ctrOnehotResultList.AddColumn(L"test target");
		ctrOnehotResultList.AddColumn(L"test output");
		ctrOnehotResultList.AddColumn(L"test accord");
	}

	m_displayWnd.GetAnalysisWnd().ReadySimulation(*m_setup_info);
}

#include "util/FileUtil.h"

simulate::Simulator* SimulationTrainStatusControl::CreateSimulatorInstance(dp::preprocessor::InitShareObject& init_share_object
	, dp::model::ProviderModelManager& ipd
	, engine::NeuralNetworkEngine& network)
{
	std::string log_path = init_share_object.GetBaseDir();
	log_path.append("_train.log");
	m_logFileWriter._SetLogPath(util::StringUtil::MultiByteToWide(log_path).c_str());
	m_logFileWriter._SetLogSize(1024 * 1024 * 100);

	engine::TRAIN_SETUP setup;
	memset(&setup, 0, sizeof(engine::TRAIN_SETUP));
	setup.log_writer = &m_logFileWriter;

	setup.data.provider = new dp::preprocessor::DataProvider(init_share_object, m_setup_info->data_noising, m_setup_info->useNdf, m_setup_info->minibatch_size);
	setup.data.provider->SetDataSource(m_setup_info->uid_datanames_map);
	if (!setup.data.provider->Create(ipd.GetFinalProvider(false)))
	{
		delete setup.data.provider;
		return false;
	}

	setup.learn = m_setup_info->learn_info;
	if (setup.learn.learn_type == engine::_learn_type::learn_test_both)
	{
		setup.data.test_provider = new dp::preprocessor::DataProvider(init_share_object, m_setup_info->data_noising, m_setup_info->useNdf, m_setup_info->minibatch_size);
		setup.data.test_provider->SetDataSource(m_setup_info->test_uid_datanames_map);
		if (!setup.data.test_provider->Create(ipd.GetFinalProvider(false)))
		{
			delete setup.data.test_provider;
			setup.data.test_provider = NULL;

			DEBUG_OUTPUT(L"failed create providers to test after learn");

			setup.learn.learn_type = engine::_learn_type::learn;
		}
	}

	setup.recv_signal = &m_displayWnd;

	return new LearnSimulator(network, setup, m_setup_info->minibatch_size);
}

engine::_sigout SimulationTrainStatusControl::NetworkSignalProcess(_NET_SIGNAL_INFO& info)
{
	switch (info.GetType())
	{
	case _signal_type::learn_start:
		learn_start_signal((const _LEARN_START_INFO&)info);
		break;
	case _signal_type::learn_end:
		learn_end_signal((const _LEARN_END_INFO&)info);
		break;
	case _signal_type::learn_epoch_start:
		epoch_start_signal((const _EPOCH_START_INFO&)info);
		break;
	case _signal_type::learn_epoch_end:
		return epoch_end_signal((const _EPOCH_END_INFO&)info);
	}
	return engine::_sigout::sig_continue;
}

void SimulationTrainStatusControl::learn_start_signal(const _LEARN_START_INFO& info)
{
	m_displayWnd.GetAnalysisWnd().Clear();

	CGroupListCtrl& ctrOnehotResultList = m_displayWnd.GetOnehotResultListCtrl();
	ctrOnehotResultList.DeleteAllItems();

	std::wstring str = m_setup_info->learn_info.learn_type == engine::_learn_type::test ? L"test : " : L"learn : ";
	str += util::StringUtil::Format<wchar_t>(L"data[%llu]", info.learn_data.data_count);
	m_epoch_group_prop->SetName(str.c_str());

	str = util::StringUtil::Format<wchar_t>(L"count[%llu], size[%llu]", info.learn_data.batch_count, info.learn_data.batch_size);
	m_batch_prop->SetValue((variant_t)str.c_str());

	if (info.has_test)
	{
		str = util::StringUtil::Format<wchar_t>(L"Test : data[%llu]", info.test_data.data_count);
		m_test_after_learn_prop->SetName(str.c_str());
		str = util::StringUtil::Format<wchar_t>(L"count[%llu], size[%llu]", info.test_data.batch_count, info.test_data.batch_size);
		m_test_after_learn_batch_prop->SetValue((variant_t)str.c_str());
	}
}

void SimulationTrainStatusControl::learn_end_signal(const _LEARN_END_INFO& info)
{
	if (m_epoch_prop)
		m_epoch_prop->SetName(util::StringUtil::Format<wchar_t>(L"Epoch : %llu", info.epoch).c_str());
}

void SimulationTrainStatusControl::epoch_start_signal(const _EPOCH_START_INFO& info)
{
	if(m_epoch_prop)
		m_epoch_prop->SetName(util::StringUtil::Format<wchar_t>(L"Epoch : %llu", info.epoch).c_str());

	if(m_batch_lr_prop)
		m_batch_lr_prop->SetValue((variant_t)util::StringUtil::Format<wchar_t>(L"%.8f", info.learn_rate).c_str());
}

engine::_sigout SimulationTrainStatusControl::epoch_end_signal(const _EPOCH_END_INFO& info)
{
	_ANALYSIS_EPOCH_INFO cur_epoch_info;
	cur_epoch_info.epoch = &info.analysis;

	m_loss_prop->SetValue((variant_t)util::StringUtil::Format<wchar_t>(L"%.5f", info.analysis.loss).c_str());
	if (m_setup_info->learn_info.analyze.bAnalyzeArgmaxAccuracy)
	{
		m_accord_prop->SetValue((variant_t)neuro_u32(info.analysis.argmax_accord));
		m_accuracy_prop->SetValue((variant_t)util::StringUtil::Format<wchar_t>(L"%.2f %%", info.analysis.accuracy*100.f).c_str());
	}

	std::wstring elapse_str;
	if (m_setup_info->learn_info.learn_type != engine::_learn_type::test)
	{
		SimulationStatusControl::GetElapseString(info.epoch_elapse, elapse_str);
		m_epoch_elapse_prop->SetValue(elapse_str.c_str());

		SimulationStatusControl::GetElapseString(info.elapse.learn, elapse_str);
		m_backward_elapse_prop->SetValue(elapse_str.c_str());

		if (info.has_test)
		{
			cur_epoch_info.test_epoch = &info.test_after_learn_analysis;

			m_test_loss_prop->SetValue((variant_t)info.test_after_learn_analysis.loss);
			if (m_setup_info->learn_info.analyze.bAnalyzeArgmaxAccuracy)
			{
				m_test_accord_prop->SetValue((variant_t)neuro_u32(info.test_after_learn_analysis.argmax_accord));
				m_test_accuracy_prop->SetValue((variant_t)util::StringUtil::Format<wchar_t>(L"%.2f %%", info.test_after_learn_analysis.accuracy*100.f).c_str());
			}
		}
	}

	SimulationStatusControl::GetElapseString(info.elapse.batch_gen, elapse_str);
	m_read_elapse_prop->SetValue(elapse_str.c_str());
	SimulationStatusControl::GetElapseString(info.elapse.forward, elapse_str);
	m_forward_elapse_prop->SetValue(elapse_str.c_str());

	SetOnehotResultList(info.output_onehot_result_vector, info.has_test);
	m_displayWnd.GetAnalysisWnd().AddHistory(cur_epoch_info);

	if (info.epoch >= m_setup_info->learn_info.epoch_count - 1)
	{
		m_displayWnd.PostSimulationMessage(L"Closed to max epoch");
		return engine::_sigout::sig_stop;
	}

	if (m_setup_info->is_stop_below_error)
	{
		neuron_error loss = info.has_test ? info.test_after_learn_analysis.loss : info.analysis.loss;
		if (loss <= m_setup_info->below_error)
		{
			std::wstring str = util::StringUtil::Format(L"The current error[%s] is below to %s"
					, util::StringUtil::Transform<wchar_t>(loss), util::StringUtil::Transform<wchar_t>(m_setup_info->below_error));

			m_displayWnd.PostSimulationMessage(str.c_str());
			return engine::_sigout::sig_stop;
		}
	}

	return engine::_sigout::sig_continue;
}

void SimulationTrainStatusControl::SetOnehotResultList(const _output_analysis_onehot_vector& output_analysis_onehot_vector, const bool has_test_after_lean)
{
	CGroupListCtrl& ctrOnehotResultList = m_displayWnd.GetOnehotResultListCtrl();
	ctrOnehotResultList.DeleteAllItems();
	if (ctrOnehotResultList.GetItemCount() == 0)
	{
		neuro_u32 onehot_result_index = 0;
		for (neuro_u32 output = 0; output < output_analysis_onehot_vector.size(); output++)
		{
			const _output_analysys_onehot& output_analysys_onehot = output_analysis_onehot_vector[output];
			CString header; header.Format(L"%u", output_analysys_onehot.layer->m_layer.uid);	// 원래는 layer의 matrix 포인터를 넣어야 한다.
			ctrOnehotResultList.InsertGroupHeader(output, header);

			for (neuro_u32 i = 0, n = output_analysys_onehot.results.size(); i < n; i++)
				ctrOnehotResultList.InsertGroupItem(output, onehot_result_index++, util::StringUtil::Transform<wchar_t>((neuro_u32)i).c_str());
		}
	}
	neuro_u32 onehot_result_index = 0;
	for (neuro_u32 output = 0; output<output_analysis_onehot_vector.size(); output++)
	{
		const _output_analysys_onehot& output_analysys_onehot = output_analysis_onehot_vector[output];

		const bool test_result = has_test_after_lean && output_analysys_onehot.test_after_learn.size() == output_analysys_onehot.results.size();
		for (neuro_u32 i = 0, n = output_analysys_onehot.results.size(); i < n; i++, onehot_result_index++)
		{
			const _analysis_onehot& analysis = output_analysys_onehot.results[i];
			ctrOnehotResultList.SetItemText(onehot_result_index, 1, util::StringUtil::Transform<wchar_t>(analysis.target_accumulate).c_str());
			ctrOnehotResultList.SetItemText(onehot_result_index, 2, util::StringUtil::Transform<wchar_t>(analysis.out_accumulate).c_str());
			ctrOnehotResultList.SetItemText(onehot_result_index, 3, util::StringUtil::Transform<wchar_t>(analysis.accord_accumulate).c_str());

			if (test_result)
			{
				const _analysis_onehot& test_after_learn_analysis = output_analysys_onehot.test_after_learn[i];
				ctrOnehotResultList.SetItemText(onehot_result_index, 4, util::StringUtil::Transform<wchar_t>(test_after_learn_analysis.target_accumulate).c_str());
				ctrOnehotResultList.SetItemText(onehot_result_index, 5, util::StringUtil::Transform<wchar_t>(test_after_learn_analysis.out_accumulate).c_str());
				ctrOnehotResultList.SetItemText(onehot_result_index, 6, util::StringUtil::Transform<wchar_t>(test_after_learn_analysis.accord_accumulate).c_str());
			}
		}
	}
	ctrOnehotResultList.ResizeHeader();
}

void SimulationTrainStatusControl::NetworkBatchSignalProcess(const BATCH_STATUS_INFO& info)
{
	m_cur_batch_prop->SetName(util::StringUtil::Format<wchar_t>(L"Batch : %llu", info.batch_no + 1).c_str());

	if (info.GetType() == _signal_type::batch_start)
	{
		if (m_batch_lr_prop)
			m_batch_lr_prop->SetValue((variant_t)util::StringUtil::Format<wchar_t>(L"%.8f", info.learn_rate).c_str());
	}
	else
	{
		if (m_batch_loss_prop)
			m_batch_loss_prop->SetValue((variant_t)util::StringUtil::Format<wchar_t>(L"%.5f", info.loss).c_str());
	}
}
