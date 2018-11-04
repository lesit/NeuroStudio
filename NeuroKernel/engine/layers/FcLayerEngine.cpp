#include "stdafx.h"

#include "FcLayerEngine.h"

using namespace np::engine;
using namespace np::engine::layers;

FcLayerEngine* FcLayerEngine::CreateInstance(const NetworkParameter& net_param, const network::HiddenLayer& layer)
{
	return new FcLayerEngine(net_param, layer);
}

FcLayerEngine::FcLayerEngine(const NetworkParameter& net_param, const network::HiddenLayer& layer)
	:HiddenLayerEngine(net_param, layer)
{
}

FcLayerEngine::~FcLayerEngine()
{
}

neuro_u32 FcLayerEngine::Get1MultiplierSizePerBatch() const
{
	return 1;
}

bool FcLayerEngine::ForwardData(bool bTrain, const _NEURO_TENSOR_DATA& input_data, const _NEURO_TENSOR_DATA& output_data)
{
	const neuro_u32 input_length = input_data.time_length * input_data.value_size;
	if (!m_net_param.math.gemm(CblasNoTrans, CblasTrans, output_data.GetBatchSize(), output_data.value_size, input_length
		, 1.f, input_data.GetBuffer(), m_inner_data_vector[0].data.GetBuffer()
		, 0.f, output_data.GetBuffer()))
	{
		DEBUG_OUTPUT(L"failed gemm for weight");
		return false;
	}

	if (m_inner_data_vector[1].data.GetBuffer())
	{
		if (!m_net_param.math.gemm(CblasNoTrans, CblasNoTrans, output_data.GetBatchSize(), output_data.value_size, 1
			, 1.f, m_net_param.sdb.one_set_vector.GetBuffer(), m_inner_data_vector[1].data.GetBuffer()
			, 1.f, output_data.GetBuffer()))
		{
			DEBUG_OUTPUT(L"failed gemm for bias");
			return false;
		}
	}

	return true;
}

bool FcLayerEngine::BackwardError(const _NEURO_TENSOR_DATA& output_data
	, const _NEURO_TENSOR_DATA& current_error
	, const _NEURO_TENSOR_DATA& input_data
	, const _NEURO_TENSOR_DATA& input_error)
{
	const neuro_u32 input_length = input_data.time_length * input_data.value_size;
	if (!m_net_param.math.gemm(CblasNoTrans, CblasNoTrans, current_error.GetBatchSize(), input_length, current_error.value_size
		, 1.f, current_error.GetBuffer(), m_inner_data_vector[0].data.GetBuffer()
		, 0.f, input_error.GetBuffer()))
	{
		DEBUG_OUTPUT(L"failed gemm for error");
		return false;
	}
	return true;
}

bool FcLayerEngine::BackwardWeight(neuro_u32 index
	, const _NEURO_TENSOR_DATA& current_error
	, const _NEURO_TENSOR_DATA& output_data
	, const _NEURO_TENSOR_DATA& input_data
	, const _VALUE_VECTOR& grad_weight)
{
	if (index == 0)
	{
		if (!m_net_param.math.gemm(CblasTrans, CblasNoTrans, current_error.value_size, input_data.value_size, input_data.GetBatchSize()
			, 1.f, current_error.GetBuffer(), input_data.GetBuffer()
			, 0.f, grad_weight.GetBuffer()))
		{
			DEBUG_OUTPUT(L"failed gemm for weight");
			return false;
		}
	}
	else
	{
		// cublas에서는 db_minibatch 가 필요없다. 왜냐면 한번에 merge까지 해주므로..
		if (!m_net_param.math.gemv(CblasTrans, current_error.GetBatchSize(), current_error.value_size
			, 1.f, current_error.GetBuffer(), m_net_param.sdb.one_set_vector.GetBuffer()
			, 0.f, grad_weight.GetBuffer()))
		{
			DEBUG_OUTPUT(L"failed gemv for bias");
			return false;
		}
	}
	return true;
}
