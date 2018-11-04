#include "stdafx.h"

#include "NeuralNetworkPredictor.h"

#include "SharedDataBuffers.h"

using namespace np::engine;

NeuralNetworkPredictor::NeuralNetworkPredictor(NeuralNetworkEngine& nn)
: NeuralNetworkProcessor(nn)
{
}

NeuralNetworkPredictor::~NeuralNetworkPredictor()
{
}
/*
bool NeuralNetworkPredictor::Run(const _VALUE_VECTOR& input, const _VALUE_VECTOR* output)
{
	HiddenLayer* output_layer = m_network.GetOutputLayer();
	if (output_layer == NULL)
		return false;

	const _VALUE_VECTOR& output_vector = output_layer->GetOutputBuffer();
	if (output_vector.count == 0)	// 할당되지 않았다.
		return false;

	if (output != NULL && output_vector.count != output->count)
		return false;

	if (!m_network.GetInputLayer()->SetValues(input))
		return false;

	if (!Propagate(false))
		return false;

	if (output)
		return output->CopyFrom(output_vector);

	return true;
}
*/
#include "MiniBatchGenerator.h"

bool NeuralNetworkPredictor::Run(const engine::PREDICT_SETUP& setup)
{
	if (setup.provider == NULL)
		return false;

	Timer total_timer;

	const _input_engine_vector& input_engine_vector = m_network.GetInputEngineVector();
	if (input_engine_vector.size() == 0)
	{
		DEBUG_OUTPUT(L"no input");
		return false;
	}

	const _output_engine_vector& output_engine_vector = m_network.GetOutputEngineVector();
	if (output_engine_vector.size() == 0)
	{
		DEBUG_OUTPUT(L"no output");
		return false;
	}

	_producer_layer_data_vector data_vector;
	data_vector.resize(input_engine_vector.size());
	for (neuro_u32 i = 0; i < input_engine_vector.size(); i++)
	{
		InputLayerEngine* engine = input_engine_vector[i];

		_PRODUCER_LAYER_DATA_SET& producer_layer_data_set = data_vector[i];

		producer_layer_data_set.producer = FindLayerBindingProducer(*setup.provider, *engine);
		if (producer_layer_data_set.producer == NULL)
		{
			DEBUG_OUTPUT(L"no binding producer for input layer[%u]", engine->m_layer.uid);
			return false;
		}
		producer_layer_data_set.producer_dim_size = producer_layer_data_set.producer->m_data_dim_size;

		const tensor::_NEURO_TENSOR_DATA& data = engine->GetOutputData();
		producer_layer_data_set.layer_mm = &data.data.mm;
		producer_layer_data_set.layer_buffer = data.GetBuffer();
		producer_layer_data_set.layer_data_size = data.GetTimeValueSize();
	}

	_NEURO_TENSOR_DATA write_buffer(core::math_device_type::cpu, true);

	MiniBatchSequentialGenerator batch_gen(*setup.provider, data_vector);
	if (!batch_gen.Ready(setup.provider->batch_size))
	{
		DEBUG_OUTPUT(L"failed ready of minibatch generator");
		return false;
	}

	batch_gen.NewEpochStart();

	if (setup.recv_signal)
	{
		_PREDICT_START_INFO info;
		info.data.batch_size = batch_gen.GetBatchSize();
		info.data.data_count = batch_gen.GetTotalDataCount();
		info.data.batch_count = batch_gen.GetBatchCount();
		info.total_elapse = total_timer.elapsed();
		setup.recv_signal->network_signal(info);
	}

	neuro_u32 batch_count = batch_gen.GetBatchCount();

	neuro_float batch_gen_elapse = 0.f;
	neuro_float forward_elapse = 0.f;

	BATCH_STATUS_INFO batch_info;
	for (neuro_u32 batch_no = 0; batch_no < batch_count; batch_no++)
	{
		Timer batch_timer;
		neuro_u32 batch_size = batch_gen.ReadBatchData(false);
		batch_gen_elapse += batch_timer.elapsed();

		if (setup.recv_signal)
		{
			batch_info.type = _signal_type::batch_start;
			batch_info.total_elapse = total_timer.elapsed();
			batch_info.batch_no = batch_no;
			batch_info.batch_size = batch_size;
			setup.recv_signal->network_signal(batch_info);
		}

		batch_timer.restart();

		if (!Propagate(false, batch_size))
			return false;
		
		forward_elapse += batch_timer.elapsed();

		bool bRet = true;
		if (setup.result_writer)
		{
			if (output_engine_vector.size() == 1)
			{
				const _NEURO_TENSOR_DATA& tensor_data = output_engine_vector[0]->GetOutputData();
				write_buffer.AllocLike(tensor_data);
				write_buffer.CopyFrom(tensor_data);
				write_buffer.batch_size = batch_size;
				if (!setup.result_writer->Write(write_buffer))
				{
					DEBUG_OUTPUT(L"failed write result");
					bRet = false;
				}
			}
			for (neuro_size_t i = 0, n = output_engine_vector.size(); i < n; i++)
			{
			}
		}

		if (setup.recv_signal)
		{
			batch_info.type = _signal_type::batch_end;
			batch_info.total_elapse = total_timer.elapsed();
			_sigout sig_ret = setup.recv_signal->network_signal(batch_info);
			if (sig_ret != _sigout::sig_continue)
			{
				DEBUG_OUTPUT(L"user stop at batch end");
				break;
			}
		}

		if (!bRet)
			return false;

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
	if (setup.recv_signal)
	{
		_PREDICT_END_INFO info;
		info.batch_gen_elapse = batch_gen_elapse;
		info.forward_elapse = forward_elapse;
		info.total_elapse = total_timer.elapsed();
		setup.recv_signal->network_signal(info);
	}

	return true;
}
