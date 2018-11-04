#include "stdafx.h"

#include "DropouLayerCpuEngine.h"

#include "util/cpu_parallel_for.h"

#include "util/randoms.h"

using namespace np::engine;
using namespace np::engine::layers;

DropouLayerCpuEngine::DropouLayerCpuEngine(const NetworkParameter& net_param, const network::HiddenLayer& layer)
: DropoutLayerEngine(net_param, layer)
{
}


DropouLayerCpuEngine::~DropouLayerCpuEngine()
{
	m_mask.Dealloc();
}

bool DropouLayerCpuEngine::ForwardData(bool bTrain, const _NEURO_TENSOR_DATA& input_data, const _NEURO_TENSOR_DATA& output_data)
{
	if (output_data.GetSize() != input_data.GetSize())
	{
		DEBUG_OUTPUT(L"output data's size is not equal with input");
		return false;
	}

	neuro_size_t in_size = input_data.value_size;

	bool bRet = true;
	for_i(input_data.GetBatchTimeSize(), [&](neuro_u32 sample)
	{
		const neuron_value* in_ptr = input_data.GetBatchTimeData(sample);
		neuron_value* out_ptr = output_data.GetBatchTimeData(sample);
		// 이상하다!!
		if (in_ptr == NULL || out_ptr == NULL)
		{
			DEBUG_OUTPUT(L"in/out GetBuffer() pointer is null");
			bRet = false;
			return;
		}

		neuro_u32* mask = m_mask.GetBatchTimeData(sample);

		if (bTrain)
		{
			for (neuro_size_t i = 0; i < in_size; i++)
			{
				mask[i] = bernoulli(m_entry.dropout.dropout_rate);
				out_ptr[i] = in_ptr[i] * mask[i] * m_dropout_scale;
			}
		}
		else
		{
			memcpy(out_ptr, in_ptr, sizeof(neuron_value)*in_size);
		}
	});

	return bRet;
}

bool DropouLayerCpuEngine::BackwardError(const _NEURO_TENSOR_DATA& output_data
	, const _NEURO_TENSOR_DATA& current_error
	, const _NEURO_TENSOR_DATA& input_data
	, const _NEURO_TENSOR_DATA& input_error)
{
	if (m_mask.GetBuffer() == NULL)
	{
		DEBUG_OUTPUT(L"size of mask is not equal with output");
		return false;
	}

	bool bRet = true;

	for_i(current_error.GetBatchTimeSize(), [&](neuro_u32 sample)
	{
		neuro_u32* mask = m_mask.GetBatchTimeData(sample);

		neuron_value* out_error = current_error.GetBatchTimeData(sample);
		neuron_value* in_error = input_error.GetBatchTimeData(sample);
		if (!out_error || !in_error)
		{
			// 전파할 에러가 없다!
			DEBUG_OUTPUT(L"no input error GetBuffer() pointer");

			bRet = false;
			return;
		}

		for (size_t i = 0; i < current_error.value_size; i++)
			in_error[i] = mask[i] * out_error[i];
	});

	if (!bRet)
	{
		DEBUG_OUTPUT(L"failed backward");
		return false;
	}
	return true;
}
