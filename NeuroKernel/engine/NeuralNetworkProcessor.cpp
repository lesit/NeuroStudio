#include "stdafx.h"

#include "NeuralNetworkProcessor.h"
#include "SharedDataBuffers.h"

using namespace np::engine;

NeuralNetworkProcessor::NeuralNetworkProcessor(NeuralNetworkEngine& engine)
: m_network(engine), m_uid_layer_map(engine.GetUidEngineMap()), m_engine_vector(m_network.GetHiddenEngineVector())
{
}

NeuralNetworkProcessor::~NeuralNetworkProcessor()
{
}

AbstractProducer* NeuralNetworkProcessor::FindLayerBindingProducer(DataProvider& provider, const AbstractLayerEngine& engine)
{
	const _neuro_binding_model_set& binding_model_set = engine.m_layer.GetBindingSet();
	_neuro_binding_model_set::const_iterator it = binding_model_set.begin();
	for (; it != binding_model_set.end(); it++)
	{
		NetworkBindingModel* binding = *it;
		if (binding->GetBindingModelType() == _binding_model_type::data_producer)
		{
			AbstractProducer* producer = provider.FindBindingProducer(binding);
			if (producer)
				return producer;
		}
	}
	return NULL;
}

bool NeuralNetworkProcessor::Ready(neuro_u32 batch_size)
{
	return m_network.AllocBuffers(batch_size);
}

bool NeuralNetworkProcessor::Propagate(bool bTrain, neuro_u32 batch_size)
{
	layers::_hidden_engine_vector::const_iterator it = m_engine_vector.begin(), end = m_engine_vector.end();
	for (; it != end; it++)
	{
		layers::HiddenLayerEngine* layer = *it;
		if (!layer->Propagation(bTrain, batch_size))
		{
			DEBUG_OUTPUT(L"propagation. failed propagation. %u th layer", it - m_engine_vector.begin());
			return false;
		}
	}
	return true;
}
