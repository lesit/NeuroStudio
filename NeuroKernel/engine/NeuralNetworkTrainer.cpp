#include "stdafx.h"

#include "NeuralNetworkTrainer.h"

#include "SharedDataBuffers.h"
#include "TrainFunctions.h"

#include "WeightStoreManager.h"

using namespace np::engine;

NeuralNetworkTrainer::NeuralNetworkTrainer(NeuralNetworkEngine& nn)
: NeuralNetworkProcessor(nn)
{
	m_analyze_loss = NULL;
}

NeuralNetworkTrainer::~NeuralNetworkTrainer()
{
	m_cpu_out_buffer.Dealloc();
	m_cpu_target_buffer.Dealloc();

	delete m_analyze_loss;
	m_analyze_loss = NULL;
}

bool NeuralNetworkTrainer::Ready(neuro_u32 batch_size)
{
	if (!__super::Ready(batch_size))
		return false;

	return true;
}

bool NeuralNetworkTrainer::Train(const TRAIN_SETUP& setup)
{
	np::Timer total_timer;

	const _input_engine_vector& input_engine_vector = m_network.GetInputEngineVector();
	if (input_engine_vector.size() == 0)
	{
		DEBUG_OUTPUT(L"no input");
		return false;
	}
	const _output_engine_vector& output_engine_vector = m_network.GetOutputEngineVector();
	if (output_engine_vector.size() == 0)
	{
		DEBUG_OUTPUT(L"no output buffer");
		return false;
	}

	_producer_layer_data_vector data_vector;
	data_vector.resize(input_engine_vector.size()+ output_engine_vector.size());
	_producer_layer_data_vector test_data_vector;
	test_data_vector.resize(data_vector.size());

	_producer_layer_data_vector::iterator it_data = data_vector.begin();
	_producer_layer_data_vector::iterator it_test_data = test_data_vector.begin();

	// test batch generator에 data.provider가 아닌 data.test_provider 의 producer를 넣어야 한다!
	for (neuro_u32 i = 0; i < input_engine_vector.size(); i++)
	{
		InputLayerEngine* engine = input_engine_vector[i];

		_PRODUCER_LAYER_DATA_SET& data_set = *(it_data++);

		data_set.producer = FindLayerBindingProducer(*setup.data.provider, *engine);
		if (data_set.producer == NULL)
		{
			DEBUG_OUTPUT(L"no binding producer for input layer[%u]", engine->m_layer.uid);
			return false;
		}
		data_set.producer_dim_size = data_set.producer->m_data_dim_size;

		const tensor::_NEURO_TENSOR_DATA& data = engine->GetOutputData();
		data_set.layer_mm = &data.data.mm;
		data_set.layer_buffer = data.GetBuffer();
		data_set.layer_data_size = data.GetTimeValueSize();

		if (setup.data.test_provider)
		{
			_PRODUCER_LAYER_DATA_SET& test_data_set = *(it_test_data++);
			test_data_set = data_set;
			test_data_set.producer = FindLayerBindingProducer(*setup.data.test_provider, *engine);
		}
	}
	for (neuro_u32 i = 0; i < output_engine_vector.size(); i++)
	{
		OutputLayerEngine* engine = output_engine_vector[i];

		_PRODUCER_LAYER_DATA_SET& data_set = *(it_data++);

		data_set.producer = FindLayerBindingProducer(*setup.data.provider, *engine);
		if (data_set.producer == NULL)
		{
			DEBUG_OUTPUT(L"no binding producer for output layer[%u]", engine->m_layer.uid);
			return false;
		}
		data_set.producer_dim_size = data_set.producer->m_data_dim_size;

		data_set.read_label = data_set.producer->m_label_out_type!=dp::model::_label_out_type::none;

		const tensor::_TYPED_TENSOR_DATA<void*, 4>& data = engine->GetTargetData();
		data_set.layer_mm = &data.data.mm;
		data_set.layer_buffer = data.GetBuffer();
		data_set.layer_data_size = data.GetTimeValueSize();

		if (setup.data.test_provider)
		{
			_PRODUCER_LAYER_DATA_SET& test_data_set = *(it_test_data++);
			test_data_set = data_set;
			test_data_set.producer = FindLayerBindingProducer(*setup.data.test_provider, *engine);
		}
	}

	_learn_type learn_type = setup.learn.learn_type;
	const bool isLearnMode = learn_type != _learn_type::test;
	const bool isWeightInit = isLearnMode && setup.learn.isWeightInit;

	std::vector<neuro_float> opt_parameters;
	if (isWeightInit)
		DEBUG_OUTPUT(L"init weights");
	else
		opt_parameters = m_network.GetOptimizerParameters();

	TrainFunctions train_funcs;
	if (!train_funcs.Initialize(m_network.GetNetParam().run_pdtype, m_network.GetNetParam().cuda_instance
		, m_network.GetLearningInfo(), opt_parameters
		, *setup.data.provider, data_vector
		, setup.data.test_provider, test_data_vector))
	{
		DEBUG_OUTPUT(L"failed initialize train functions");
		return false;
	}
	m_network.SetOptimizer(train_funcs.optimize_epoch);

	neuron_error last_snapshot_loss = m_network.GetLearnHistory()->last_loss;
	if (last_snapshot_loss == 0.f)
		last_snapshot_loss = 100;

	neuro_float last_snapshot_accuracy = m_network.GetLearnHistory()->last_accuracy;

	if (learn_type == _learn_type::learn_test_both && train_funcs.test_batch_gen == NULL)
		learn_type = _learn_type::learn;

	WeightStoreManager& wsm = m_network.GetWeightStoreManager();

	bool retInitWeights = wsm.InitAllWeights(isWeightInit, train_funcs.optimize_epoch->GetOptimizer()->GetHistoryCount());
	if (isWeightInit)
	{
		m_network.ClearOptimizerParameters();
		m_network.SetLastLearnHistory(100, 0);
		setup.log_writer->_WriteWithTime(L"init all weights is %s\r\n", retInitWeights ? L"successed" : L"failed");
	}

	const neuro_u32 batch_size = train_funcs.batch_gen->GetBatchSize();
	const neuro_size_t batch_count = train_funcs.batch_gen->GetBatchCount();
	if (batch_count == 0)
	{
		DEBUG_OUTPUT(L"batch count in a epoch is zero");
		return false;
	}

	if (setup.log_writer)
	{
		std::wstring title = L"------ start of training : ";
		std::wstring loss;
		for (neuro_u32 i = 0; i < output_engine_vector.size(); i++)
		{
			OutputLayerEngine* layer = output_engine_vector[i];
			network::_loss_type type = (network::_loss_type)layer->GetEntry().output.loss_type;
			loss.append(ToString(type));
			loss.append(L",");
		}

		if (!loss.empty())
		{
			loss.erase(loss.end() - 1);
			title.append(L"loss=\"").append(loss).append(L"\", ");
		}
		if (train_funcs.optimize_epoch)
		{
			if (train_funcs.optimize_epoch->GetOptimizer())
			{
				title.append(L"optimizer=\"").append(ToString(train_funcs.optimize_epoch->GetOptimizer()->type())).append(L"\", ");
			};
			title.append(L"opt_policy=\"").append(ToString(train_funcs.optimize_epoch->GetRule().lr_policy.type)).append(L"\", ");
			title.append(L"opt_gama=").append(util::StringUtil::Transform<wchar_t>(train_funcs.optimize_epoch->GetRule().lr_policy.gamma)).append(L", ");
			title.append(L"opt_step=").append(util::StringUtil::Transform<wchar_t>(train_funcs.optimize_epoch->GetRule().lr_policy.step)).append(L", ");
		}
		title.append(util::StringUtil::Format(L"data=%llu, batch count=%llu, batch size=%u\r\n"
			, train_funcs.batch_gen->GetTotalDataCount(), batch_count, batch_size));

		setup.log_writer->_WriteWithTime(title.c_str());
	}

	if (m_analyze_loss == NULL)
		m_analyze_loss = new AnalysisLoss();
	m_analyze_loss->condition = setup.learn.analyze;

	if (setup.recv_signal)
	{
		_LEARN_START_INFO info;

		info.learn_data.batch_size = train_funcs.batch_gen->GetBatchSize();
		info.learn_data.data_count = train_funcs.batch_gen->GetTotalDataCount();
		info.learn_data.batch_count = train_funcs.batch_gen->GetBatchCount();

		if (train_funcs.test_batch_gen)
		{
			info.has_test = true;
			info.test_data.batch_size = train_funcs.test_batch_gen->GetBatchSize();
			info.test_data.data_count = train_funcs.test_batch_gen->GetTotalDataCount();
			info.test_data.batch_count = train_funcs.test_batch_gen->GetBatchCount();
		}
		else
			info.has_test = false;
		info.total_elapse = total_timer.elapsed();
		setup.recv_signal->network_signal(info);
	}

	auto store_net_parameters = [&]()
	{
		if (!wsm.UpdateWeights())
			return false;

		opt_parameters = train_funcs.optimize_epoch->GetOptimizer()->GetParameters();
		m_network.SetOptimizerParameters(opt_parameters);
		m_network.SaveRootEntry();
		return true;
	};

	_output_analysis_onehot_vector output_onehot_result_vector;
	_output_analysys_onehot_map output_analysys_onehot_map;

	neuro_u32 epoch = 0;
	neuro_size_t epoch_count = isLearnMode ? setup.learn.epoch_count : 1;
	for (; epoch < epoch_count; epoch++)
	{
		const neuro_float learn_rate = train_funcs.optimize_epoch->GetLearnRate();
		if (setup.recv_signal)
		{
			_EPOCH_START_INFO info;
			info.epoch = epoch;
			info.learn_rate = learn_rate;
			info.total_elapse = total_timer.elapsed();

			std::unordered_set<neuro_u32> onehot_result_output_set;
			if (setup.recv_signal->network_signal(info, &onehot_result_output_set) != _sigout::sig_continue)
				break;

			// 다하는게 아니라 선택한 것만 한다!
			output_onehot_result_vector.clear();
			output_analysys_onehot_map.clear();
			for (neuro_u32 i = 0; i < output_engine_vector.size(); i++)
			{
				OutputLayerEngine* engine = output_engine_vector[i];
				if (onehot_result_output_set.find(engine->m_layer.uid) == onehot_result_output_set.end())
					continue;

				output_onehot_result_vector.resize(output_onehot_result_vector.size() + 1);

				_output_analysys_onehot& result = output_onehot_result_vector.back();
				result.layer = engine;

				neuro_u32 size = engine->GetOutputData().value_size;
				result.results.resize(size);

				if (train_funcs.test_batch_gen)
					result.test_after_learn.resize(size);

				output_analysys_onehot_map[engine->m_layer.uid] = &result;
			}
		}

		if (learn_rate < optimizer::end_learn_rate)
		{
			DEBUG_OUTPUT(L"learn rate[%.20f] is under %f. stop learning", learn_rate, optimizer::end_learn_rate);

			if (setup.log_writer)
				setup.log_writer->_WriteWithTime(L"epoch[%llu] learn rate[%.20f] is under %f. stop learning\r\n", epoch, learn_rate, optimizer::end_learn_rate);
			break;
		}

		DEBUG_OUTPUT(L"");
		DEBUG_OUTPUT(L"epoch:%llu", epoch);

		_EPOCH_END_INFO epoch_end_info;
		memset(&epoch_end_info.elapse, 0, sizeof(LEARN_EPOCH_ELAPSE));
		epoch_end_info.epoch = epoch;
		epoch_end_info.has_test = false;

		np::Timer epoch_timer;

		_sigout sig_ret;
		if(! RunEpoch(total_timer
			, isLearnMode, *train_funcs.optimize_epoch
			, *train_funcs.batch_gen
			, epoch
			, epoch_end_info.analysis
			, &epoch_end_info.elapse
			, output_analysys_onehot_map, false
			, setup.recv_signal, &sig_ret))
		{
			if (setup.log_writer)
				setup.log_writer->_WriteWithTime(L"failed run epoch[%llu]\r\n", epoch);
			return false;
		}
		if (sig_ret != _sigout::sig_continue)
		{
			DEBUG_OUTPUT(L"stopped after RunEpoch");
			++epoch;
			break;
		}

		if (train_funcs.test_batch_gen)	// 어떤지 테스트셋으로 테스트 해보자!!
		{
			epoch_end_info.has_test = true;
			if(!RunEpoch(total_timer
				, false, *train_funcs.optimize_epoch
				, *train_funcs.test_batch_gen
				, epoch
				, epoch_end_info.test_after_learn_analysis
				, NULL
				, output_analysys_onehot_map, true
				, NULL, NULL))
			{
				DEBUG_OUTPUT(L"failed test after learn");
				if (setup.log_writer)
					setup.log_writer->_WriteWithTime(L"failed run test epoch[%llu]\r\n", epoch);

				epoch_end_info.has_test = false;
			}
		}

		if (isLearnMode)
		{
			analysis_loss::_epoch_info* cur_analysis_epoch_info = epoch_end_info.has_test ? &epoch_end_info.test_after_learn_analysis : &epoch_end_info.analysis;

//			if (last_snapshot_accuracy < cur_analysis_epoch_info->accuracy)	// 가짜뉴스 찾기용
			if (last_snapshot_loss > cur_analysis_epoch_info->loss
				|| last_snapshot_accuracy < cur_analysis_epoch_info->accuracy)
			{
				last_snapshot_loss = cur_analysis_epoch_info->loss;
				last_snapshot_accuracy = cur_analysis_epoch_info->accuracy;

				if (!wsm.WeightsToSnapshot())
				{
					DEBUG_OUTPUT(L"failed snapshot weights");
					if (setup.log_writer)
						setup.log_writer->_WriteWithTime(L"failed snapshot weights. epoch[%llu]\r\n", epoch + 1);
					return false;
				}
				m_network.SetLastLearnHistory(last_snapshot_loss, last_snapshot_accuracy);
			}
			if ((epoch + 1) % 10 == 0)
			{
				if (!store_net_parameters())
				{
					DEBUG_OUTPUT(L"failed update weights");
					if (setup.log_writer)
						setup.log_writer->_WriteWithTime(L"failed upate weights. epoch[%llu]\r\n", epoch + 1);
					return false;
				}
				m_network.SaveRootEntry();
			}
			train_funcs.optimize_epoch->NextEpoch();
		}

		epoch_end_info.epoch_elapse = epoch_timer.elapsed();
		if (setup.log_writer)
		{
			const wchar_t* run_title = isLearnMode ? L"learning" : L"test";

			std::wstring result = util::StringUtil::Format(L"epoch[%08llu] lr[%f ~ %f] - ", epoch, learn_rate, train_funcs.optimize_epoch->GetLearnRate());
			result.append(run_title);
			result.append(util::StringUtil::Format(L" : loss=%f, accuracy=%f", epoch_end_info.analysis.loss, epoch_end_info.analysis.accuracy));
			if (epoch_end_info.has_test)
				result.append(util::StringUtil::Format(L", test : loss=%f, accuracy=%f"
					, epoch_end_info.test_after_learn_analysis.loss, epoch_end_info.test_after_learn_analysis.accuracy));
			result.append(L"\r\n");

			if (output_onehot_result_vector.size()>0)
			{
				result.append(L"onehot results -->\r\n");
				result.append(run_title).append(util::StringUtil::Format(L" : total %llu\r\n", epoch_end_info.analysis.data_count));
				MakeEpochOnehotResult(output_onehot_result_vector, result);
				result.append(L"<- onehot results\r\n\r\n");
			}
			setup.log_writer->_WriteWithTime(result.c_str());
		}

		if (setup.recv_signal)
		{
			epoch_end_info.output_onehot_result_vector = output_onehot_result_vector;
			epoch_end_info.total_elapse = total_timer.elapsed();
			_sigout epoch_sig_ret = setup.recv_signal->network_signal(epoch_end_info);

			if (epoch_sig_ret != _sigout::sig_continue)
			{
				DEBUG_OUTPUT(L"user stop at epoch end");
				sig_ret = epoch_sig_ret;
			}
		}

		if (sig_ret != _sigout::sig_continue)
		{
			DEBUG_OUTPUT(L"stopped after epoch_end_signal");
			++epoch;
			break;
		}
	}

	if (setup.recv_signal)
	{
		_LEARN_END_INFO info;
		info.epoch = epoch;
		info.total_elapse = total_timer.elapsed();
		setup.recv_signal->network_signal(info);
	}

	DEBUG_OUTPUT(L"end. epoch=%llu\r\n", epoch);

	if (setup.log_writer)
		setup.log_writer->_WriteWithTime(L"<---- end of training ----- last epoch[%llu]\r\n\r\n", epoch);

	if (isLearnMode)
	{
		if (!store_net_parameters())
		{
			DEBUG_OUTPUT(L"failed update weights");
			if (setup.log_writer)
				setup.log_writer->_WriteWithTime(L"failed upate weights\r\n");
			return false;
		}
		wsm.SnapshotToWeights();	// 다시 시작할때 사용하기 위해서 마지막에 snapshot했던 것을 다시 실제 weight 버퍼로 옮겨놓는다.
	}

	return true;
}

bool NeuralNetworkTrainer::RunEpoch(np::Timer& total_timer, bool is_learn_mode, optimizer::OptimizeInEpoch& optimize_epoch
	, MiniBatchGenerator& batch_gen
	, neuro_u32 epoch
	, analysis_loss::_epoch_info& analysis_epoch_info
	, LEARN_EPOCH_ELAPSE* learn_elapse
	, _output_analysys_onehot_map& output_analysys_onehot_map
	, bool is_test_output_analysis
	, RecvSignal* recv_batch_signal, _sigout* sig_ret)
{
	analysis_epoch_info.data_count = 0;

	if (!batch_gen.NewEpochStart())
	{
		DEBUG_OUTPUT(L"no new epoch");
		return false;
	}

	const engine::_output_engine_vector& output_engine_vector = m_network.GetOutputEngineVector();

	if (sig_ret)
		*sig_ret = _sigout::sig_continue;

	_output_analysys_onehot_map::iterator it_result_analysis = output_analysys_onehot_map.begin();
	for (; it_result_analysis!= output_analysys_onehot_map.end(); it_result_analysis++)
	{
		_output_analysys_onehot& output_analysys_onehot = *it_result_analysis->second;

		if(is_test_output_analysis)
			memset(output_analysys_onehot.test_after_learn.data(), 0, sizeof(_analysis_onehot) * output_analysys_onehot.test_after_learn.size());
		else
			memset(output_analysys_onehot.results.data(), 0, sizeof(_analysis_onehot) * output_analysys_onehot.results.size());
	}

	neuro_float epoch_loss = 0;

	neuro_u32 original_batch_size = batch_gen.GetBatchSize();
	neuro_u32 batch_count = batch_gen.GetBatchCount();

	BATCH_STATUS_INFO batch_info;
	for (neuro_u32 batch_no=0; batch_no < batch_count; batch_no++)
	{
#ifdef _DEBUG
		if (batch_no == batch_count - 1)
			int a = 0;
#endif
		Timer timer;

		neuro_u32 read_batch_size = batch_gen.ReadBatchData(is_learn_mode);
		if (recv_batch_signal)
		{
			batch_info.type = _signal_type::batch_start;
			batch_info.total_elapse = total_timer.elapsed();
			batch_info.batch_no = batch_no;
			batch_info.batch_size = read_batch_size;
			batch_info.learn_rate = optimize_epoch.GetLearnRate();
			recv_batch_signal->network_signal(batch_info);
		}

		if (read_batch_size == 0)
		{
			DEBUG_OUTPUT(L"failed read input and target batch data from batch generator");
			return 0;
		}
		else if (is_learn_mode && read_batch_size != original_batch_size)	// 이럴리가 없다. ReadBatchData에서 그렇게 안함
		{
			DEBUG_OUTPUT(L"read size[%u] is not batch size[%u]", read_batch_size, original_batch_size);
			return false;
		}

		analysis_epoch_info.data_count += neuro_size_t(read_batch_size);

		if (learn_elapse)
			learn_elapse->batch_gen += neuro_float(timer.elapsed());

		timer.restart();
		if (!Propagate(is_learn_mode, read_batch_size))
		{
			DEBUG_OUTPUT(L"failed propagate");
			return false;
		}

		if (learn_elapse)
			learn_elapse->forward += neuro_float(timer.elapsed());

		if (is_learn_mode)
		{
			timer.restart();
			if (!Learn(read_batch_size))
			{
				DEBUG_OUTPUT(L"failed to learn");
				return false;
			}

			if (learn_elapse)
				learn_elapse->learn += neuro_float(timer.elapsed());

			optimize_epoch.NextBatch();
		}

		const neuro_float batch_loss = SumOutputLayerLoss();
		epoch_loss += batch_loss;

		neuro_u32 total_argmax_accord_count = 0;
		// collect according accumulate count
		// m_analyze_loss->condition.bAnalyzeArgmaxAccuracy 만 있다면, 오로지 accord와 accuracy를 구하기 위한 것이다!
		if (output_analysys_onehot_map.size() > 0 || m_analyze_loss->condition.bAnalyzeArgmaxAccuracy)
		{
			for (neuro_size_t i = 0; i < output_engine_vector.size(); i++)
			{
				OutputLayerEngine* engine = output_engine_vector[i];

				bool is_target_label = ((network::OutputLayer&)engine->m_layer).ReadLabelForTarget();

				const tensor::_NEURO_TENSOR_DATA& output = engine->GetOutputData();
				const tensor::_TYPED_TENSOR_DATA<void*, 4>& target = engine->GetTargetData();

				m_cpu_out_buffer.AllocLike(output);
				m_cpu_out_buffer.batch_time_order = output.batch_time_order;
				m_cpu_out_buffer.CopyFrom(output);

				m_cpu_target_buffer.AllocLike(target);
				m_cpu_target_buffer.batch_time_order = target.batch_time_order;
				m_cpu_target_buffer.CopyFrom(target);

				_analysis_onehot* analysis_onehot_vector = NULL;
				{
					_output_analysys_onehot_map::iterator it_result_analysis = output_analysys_onehot_map.find(engine->m_layer.uid);
					if (it_result_analysis != output_analysys_onehot_map.end())
						analysis_onehot_vector = is_test_output_analysis ? it_result_analysis->second->test_after_learn.data() : it_result_analysis->second->results.data();
				}

				for (neuro_u32 sample = 0; sample < read_batch_size; sample++)
				{
					neuro_size_t output_max = max_index(m_cpu_out_buffer.GetBatchData(sample), m_cpu_out_buffer.value_size);
					neuro_size_t target_max;
					if (is_target_label)
						target_max = *(neuro_u32*)m_cpu_target_buffer.GetBatchData(sample);
					else
						target_max = max_index((neuro_float*)m_cpu_target_buffer.GetBatchData(sample), m_cpu_target_buffer.value_size);

					if (analysis_onehot_vector)
					{
						++analysis_onehot_vector[output_max].out_accumulate;
						++analysis_onehot_vector[target_max].target_accumulate;
					}

					if (output_max != target_max)
						continue;

					if (analysis_onehot_vector)
						++analysis_onehot_vector[output_max].accord_accumulate;

					++total_argmax_accord_count;
				}
			}
		}
		analysis_epoch_info.argmax_accord += total_argmax_accord_count;

		if (recv_batch_signal)
		{
			batch_info.type = _signal_type::batch_end;
			batch_info.total_elapse = total_timer.elapsed();
			batch_info.loss = batch_loss * loss::LossFunction::normalize_factor(read_batch_size);
			*sig_ret = recv_batch_signal->network_signal(batch_info);
			if (*sig_ret == _sigout::sig_stop)
			{
				DEBUG_OUTPUT(L"user stop at batch end");
				break;
			}
		}

		if (!batch_gen.NextBatch())	// 이때 남은 데이터 개수에 따라 sample가 출어들수도 있다. 이걸 감안해야한다!
		{
			++batch_no;
			if (batch_no != batch_count)
			{
				DEBUG_OUTPUT(L"no next batch. batch=%llu, total batch count=%llu", batch_no, batch_count);
				return false;
			}
			break;
		}
	}

	analysis_epoch_info.loss = epoch_loss * loss::LossFunction::normalize_factor(analysis_epoch_info.data_count);
	analysis_epoch_info.accuracy = neuro_float(analysis_epoch_info.argmax_accord) / neuro_float(analysis_epoch_info.data_count*output_engine_vector.size());

	return true;
}

bool NeuralNetworkTrainer::Learn(const neuro_u32 batch_size)
{
	_hidden_engine_vector::const_reverse_iterator it_layer = m_engine_vector.rbegin();
	_hidden_engine_vector::const_reverse_iterator end_layer = m_engine_vector.rend();

	neuro_size_t layer_index = m_engine_vector.size();
	for (; it_layer != end_layer; it_layer++)
	{
		--layer_index;

		layers::HiddenLayerEngine* engine = *it_layer;
		if (!engine->Backpropagation(batch_size))
		{
			DEBUG_OUTPUT(L"failed back propagation %u th layer", layer_index);
			return false;
		}
	}

	return true;
}

neuron_error NeuralNetworkTrainer::SumOutputLayerLoss()
{
	neuron_error original_loss = neuron_error(0);

	const _output_engine_vector& output_engine_vector = m_network.GetOutputEngineVector();
	for (size_t i = 0; i < output_engine_vector.size(); i++)
		original_loss += output_engine_vector[i]->GetLoss();

	return original_loss;
}

void NeuralNetworkTrainer::MakeEpochOnehotResult(const _output_analysis_onehot_vector& output_onehot_result_vector, std::wstring& result_status)
{
	for (neuro_u32 output = 0; output < output_onehot_result_vector.size(); output++)
	{
		const _output_analysys_onehot& output_analysys_onehot = output_onehot_result_vector[output];

		result_status += util::StringUtil::Format<wchar_t>(L"layer : uid = %u\r\n", output_analysys_onehot.layer->m_layer.uid);
		if (output_analysys_onehot.test_after_learn.size() > 0)
			result_status += L"Learn\r\n";
		for (neuro_u32 i = 0; i < output_analysys_onehot.results.size(); i++)
		{
			const _analysis_onehot& analysis_onehot = output_analysys_onehot.results[i];
			result_status += util::StringUtil::Format<wchar_t>(L"\t%u class : target %u, output %u, accord %u\r\n",
				i, analysis_onehot.target_accumulate, analysis_onehot.out_accumulate, analysis_onehot.accord_accumulate);
		}
		if (output_analysys_onehot.test_after_learn.size() > 0)
		{
			result_status += L"Test\r\n";
			for (neuro_u32 i = 0; i < output_analysys_onehot.test_after_learn.size(); i++)
			{
				const _analysis_onehot& analysis_onehot = output_analysys_onehot.test_after_learn[i];
				result_status += util::StringUtil::Format<wchar_t>(L"\t%u class : target %u, output %u, accord %u\r\n",
					i, analysis_onehot.target_accumulate, analysis_onehot.out_accumulate, analysis_onehot.accord_accumulate);
			}
		}
	}
}
