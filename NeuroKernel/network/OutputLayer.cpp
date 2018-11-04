#include "stdafx.h"

#include "OutputLayer.h"

using namespace np::network;

OutputLayer::OutputLayer(neuro_u32 uid)
: HiddenLayer(uid)
{
}

OutputLayer::~OutputLayer()
{
}

void OutputLayer::ChangedBindingDataShape()
{
	if (GetMainInput() == NULL)
		return;

	const _neuro_binding_model_set& binding_set = GetBindingSet();
	for (_neuro_binding_model_set::const_iterator it = binding_set.begin(); it != binding_set.end(); it++)\
	{
		const NetworkBindingModel* binding = *it;
		dp::model::AbstractProducerModel* producer = (dp::model::AbstractProducerModel*)binding;
		if (producer->GetBindingModelType() == _binding_model_type::data_producer)
		{
			tensor::DataShape ds;
			// 연결된 producer가 label out을 한다면, 그 갯수만큼이 layer의 출력 개수이다.
			if (producer->GetLabelOutType()!=dp::model::_label_out_type::none)
				ds = { producer->GetLabelOutCount() };
			else
				ds = producer->GetDataShape();

			GetMainInput()->layer->SetOutTensorShape(ds);
			break;
		}
	}
}

void OutputLayer::OnInsertedInput(AbstractLayer* layer)
{
	// 입력 매핑 되었을 때, binding된 target producer의 data shape를 입력 매핑된 layer에 설정하도록 하자.
	ChangedBindingDataShape();
}

bool OutputLayer::AvailableChangeActivation() const
{
	return !IsClassifyLossType();
}

_activation_type OutputLayer::GetActivation() const
{
	_loss_type loss_type = (_loss_type)m_entry.output.loss_type;

	if (loss_type == _loss_type::SigmoidCrossEntropy)
		return _activation_type::sigmoid;
	else if (loss_type == _loss_type::SoftmaxWithLoss)
		return _activation_type::softmax;

	return m_activation_type;
}

bool OutputLayer::IsClassifyLossType() const
{
	_loss_type loss_type = (_loss_type)m_entry.output.loss_type;

	return loss_type == _loss_type::CrossEntropy || loss_type == _loss_type::CrossEntropyMulticlass
		|| loss_type == _loss_type::SigmoidCrossEntropy || loss_type == _loss_type::SoftmaxWithLoss;
}

bool OutputLayer::ReadLabelForTarget() const
{
	const _neuro_binding_model_set& binding_set = GetBindingSet();
	for (_neuro_binding_model_set::const_iterator it = binding_set.begin(); it != binding_set.end(); it++)\
	{
		const NetworkBindingModel* binding = *it;
		dp::model::AbstractProducerModel* producer = (dp::model::AbstractProducerModel*)binding;
		if (producer->GetBindingModelType() == _binding_model_type::data_producer)
			return producer->GetLabelOutType() != dp::model::_label_out_type::none;
	}
	return false;
}

tensor::TensorShape OutputLayer::MakeOutTensorShape() const 
{
	return GetMainInputTs();
}
