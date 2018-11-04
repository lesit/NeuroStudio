#include "stdafx.h"

#include "RnnLayerConfigure.h"

using namespace np;
using namespace np::network;

bool RnnLayerConfigure::AvailableSetSideInput(const HiddenLayer& layer, const HiddenLayer* input) const
{
	if (input==NULL || input == layer.GetSideInput())
		return false;

	const RnnLayerConfigure& in_rnn_layer = (const RnnLayerConfigure&)*input;

	// RNN 타입이 같아야 한다.
	if (in_rnn_layer.GetLayerType() != _layer_type::rnn)
		return false;

	return input->GetEntry().rnn.type == layer.GetEntry().rnn.type;
}

tensor::TensorShape RnnLayerConfigure::MakeOutTensorShape(const HiddenLayer& layer) const
{
	const nsas::_LAYER_STRUCTURE_UNION& entry = layer.GetEntry();

	neuro_u32 time_length = entry.rnn.fix_time_length;
	if (!entry.rnn.is_non_time_input && layer.GetMainInput())
		time_length = layer.GetMainInputTs().time_length;

	return tensor::TensorShape(time_length, entry.rnn.output_count, 1, 1);
}

neuro_u32 RnnLayerConfigure::GetLayerDataInfoVector(const HiddenLayer& layer, _layer_data_info_vector& info_vector) const
{
	const nsas::_LAYER_STRUCTURE_UNION& entry = layer.GetEntry();

	const neuro_u32 gate_count = GetGateCount(entry);

	info_vector.resize(3);
	info_vector[0] = { _layer_data_type::weight, gate_count * entry.rnn.output_count * layer.GetMainInputTs().GetDimSize() };	// gate
	info_vector[1] = { _layer_data_type::bias, gate_count * entry.rnn.output_count };// bias
	info_vector[2] = { _layer_data_type::weight, gate_count * entry.rnn.output_count * entry.rnn.output_count };// hidden
	return info_vector.size();
}

neuro_u32 RnnLayerConfigure::GetGateCount(const nsas::_LAYER_STRUCTURE_UNION& entry) const
{
	switch (static_cast<network::_rnn_type>(entry.rnn.type))
	{
	case network::_rnn_type::gru:
		return 3;
	case network::_rnn_type::lstm:
		return 4;
	}
	return 0;
}
