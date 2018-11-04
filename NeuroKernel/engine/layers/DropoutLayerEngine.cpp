#include "stdafx.h"

#include "DropoutLayerEngine.h"
#include "../layers/DropouLayerCpuEngine.h"
#include "../layers/DropoutLayerCudaEngine.h"

#include "util/randoms.h"

using namespace np::engine;
using namespace np::engine::layers;

DropoutLayerEngine* DropoutLayerEngine::CreateInstance(const NetworkParameter& net_param, const network::HiddenLayer& layer)
{
	if (net_param.run_pdtype == core::math_device_type::cuda)
		return new DropoutLayerCudaEngine(net_param, layer);
	else
		return new DropouLayerCpuEngine(net_param, layer);
}

DropoutLayerEngine::DropoutLayerEngine(const NetworkParameter& net_param, const network::HiddenLayer& layer)
: HiddenLayerEngine(net_param, layer), m_mask(net_param.run_pdtype, true)
{
}


DropoutLayerEngine::~DropoutLayerEngine()
{
}

bool DropoutLayerEngine::OnInitialized()
{
	if (m_out_ts.GetDimSize() != m_in_ts.GetDimSize())
	{
		DEBUG_OUTPUT(L"in/out tensor is different");
		return false;
	}

	// to avoid zero - division
	if (m_entry.dropout.dropout_rate<0 || m_entry.dropout.dropout_rate > neuro_float(0.9))
	{
		DEBUG_OUTPUT(L"dropout rate[%f] must be 0 ~ 1", m_entry.dropout.dropout_rate);
		return false;
	}

	m_dropout_scale = neuro_float(1) / (neuro_float(1) - m_entry.dropout.dropout_rate);
	m_uint_threshold = static_cast<unsigned int>(UINT_MAX * m_entry.dropout.dropout_rate);
	return true;
}

bool DropoutLayerEngine::OnOutputBufferInitialized(const _NEURO_TENSOR_DATA& buf)
{
	return m_mask.Alloc(buf.GetBatchTimeSize(), buf.value_size) != NULL;
}
