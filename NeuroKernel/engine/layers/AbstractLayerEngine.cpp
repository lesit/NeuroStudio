#include "stdafx.h"

#include "AbstractLayerEngine.h"

using namespace np::engine;
using namespace np::engine::layers;

AbstractLayerEngine::AbstractLayerEngine(const NetworkParameter& net_param, const network::AbstractLayer& layer)
: m_net_param(net_param)
, m_layer(layer)
, m_out_ts(layer.GetOutTensorShape())
{
	m_output.Dealloc();
}

AbstractLayerEngine::~AbstractLayerEngine()
{
}

bool AbstractLayerEngine::AllocOutputBuffer(core::math_device_type pdtype, neuro_u32 batch_size)
{
	if (m_output.data.mm.GetType() != pdtype)
	{
		m_output.Dealloc();
		m_output = _NEURO_TENSOR_DATA(pdtype);
	}
	m_output.Calloc(batch_size, m_out_ts.time_length, m_out_ts.GetDimSize());
	m_output.batch_time_order = TensorBatchTimeOrder();

	if (!OnOutputBufferInitialized(m_output))
	{
		DEBUG_OUTPUT(L"failed alloc ouput buffer of layer");
		return false;
	}
	return true;
}

void AbstractLayerEngine::DeallocOutputBuffer()
{
	m_output.Dealloc();
}

InputLayerEngine::InputLayerEngine(const NetworkParameter& net_param, const network::InputLayer& layer)
: AbstractLayerEngine(net_param, layer)
{
}
