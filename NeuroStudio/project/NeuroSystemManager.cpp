#include "stdafx.h"

#include "NeuroSystemManager.h"

#include "storage/FileDeviceAdaptor.h"
#include "util/FileUtil.h"

#include "NeuroStudioProject.h"

//#include "SimDefinition.h"

using namespace np::network;
using namespace np::project;

NeuroSystemManager::NeuroSystemManager()
{
	m_network = new network::NeuralNetwork;
}

NeuroSystemManager::~NeuroSystemManager()
{
	CloseAll();
}

#include "util/StringUtil.h"
#include "NeuroKernel/network/OutputLayer.h"
bool NeuroSystemManager::SampleCreate()
{
	if(!m_network)
		return false;
/*
	dp::model::AbstractReaderModel* reader1 = m_provider.GetLearnProvider().AddReaderModel(dp::model::_reader_type::text);
	dp::model::AbstractReaderModel* reader2 = m_provider.GetLearnProvider().AddReaderModel(dp::model::_reader_type::text);
	dp::model::AbstractProducerModel* producer1 = m_provider.GetLearnProvider().AddProducerModel(dp::model::_producer_type::numeric);
	dp::model::AbstractProducerModel* producer2 = m_provider.GetLearnProvider().AddProducerModel(dp::model::_producer_type::numeric);
	dp::model::AbstractProducerModel* producer3 = m_provider.GetLearnProvider().AddProducerModel(dp::model::_producer_type::numeric);

	producer1->SetInput(reader1);
	producer2->SetInput(reader1);
	producer3->SetInput(reader2);

	InputLayer* input = (InputLayer*)m_network->AddLayer(network::_layer_type::input);
	HiddenLayer* hidden1 = (HiddenLayer*)m_network->AddLayer(network::_layer_type::concat);
	HiddenLayer* hidden2 = (HiddenLayer*)m_network->AddLayer(network::_layer_type::concat);
	HiddenLayer* hidden3 = (HiddenLayer*)m_network->AddLayer(network::_layer_type::concat);
	HiddenLayer* hidden4 = (HiddenLayer*)m_network->AddLayer(network::_layer_type::concat);
	HiddenLayer* hidden5 = (HiddenLayer*)m_network->AddLayer(network::_layer_type::concat);
	HiddenLayer* hidden6 = (HiddenLayer*)m_network->AddLayer(network::_layer_type::concat);
	HiddenLayer* hidden7 = (HiddenLayer*)m_network->AddLayer(network::_layer_type::concat);
	HiddenLayer* hidden8 = (HiddenLayer*)m_network->AddLayer(network::_layer_type::concat);
	
	OutputLayer* output = (OutputLayer*)m_network->AddLayer(network::_layer_type::output);

	hidden1->InsertInput(input);
	hidden2->InsertInput(input);
	hidden3->InsertInput(hidden1);
	hidden4->InsertInput(hidden2);
	hidden5->InsertInput(hidden2);
	hidden6->InsertInput(hidden3);
	hidden8->InsertInput(hidden2);
	output->InsertInput(hidden8);

	input->AddBinding(producer1);
	output->AddBinding(producer3);
*/
	dp::model::AbstractProducerModel* learn_input_producer = m_provider.GetLearnProvider()->AddProducerModel(dp::model::_producer_type::mnist_img);
	dp::model::AbstractProducerModel* learn_target_producer = m_provider.GetLearnProvider()->AddProducerModel(dp::model::_producer_type::mnist_label);
	dp::model::AbstractProducerModel* predict_producer = m_provider.GetPredictProvider().AddProducerModel(dp::model::_producer_type::image_file);

	InputLayer* input = (InputLayer*)m_network->AddLayer(network::_layer_type::input);
	HiddenLayer* hidden1 = (HiddenLayer*)m_network->AddLayer(network::_layer_type::convolutional);
	HiddenLayer* hidden2 = (HiddenLayer*)m_network->AddLayer(network::_layer_type::pooling);
	HiddenLayer* hidden3 = (HiddenLayer*)m_network->AddLayer(network::_layer_type::convolutional);
	HiddenLayer* hidden4 = (HiddenLayer*)m_network->AddLayer(network::_layer_type::pooling);
	HiddenLayer* hidden5 = (HiddenLayer*)m_network->AddLayer(network::_layer_type::dropout);
	OutputLayer* output = (OutputLayer*)m_network->AddLayer(network::_layer_type::output);

	hidden1->InsertInput(input);
	hidden2->InsertInput(hidden1);
	hidden3->InsertInput(hidden2);
	hidden4->InsertInput(hidden3);
	hidden5->InsertInput(hidden4);
	output->InsertInput(hidden5);

	input->AddBinding(learn_input_producer);
	output->AddBinding(learn_target_producer);
	input->AddBinding(predict_producer);

	return true;
}

void NeuroSystemManager::NewSystem()
{
	CloseAll();

	NetworkNew();
}

void NeuroSystemManager::CloseAll()
{
	if (m_network)
		delete m_network;
	m_network = NULL;

	m_provider.ClearAll();
}

bool NeuroSystemManager::NetworkNew()
{
	if (m_network)
		delete m_network;

	m_network = new network::NeuralNetwork;
	return m_network != NULL;
}

bool NeuroSystemManager::NetworkLoad(device::IODeviceFactory* nd_desc)
{
	if (!nd_desc)
		return false;

	if (!NetworkNew())
		return false;

	if(!m_network->Load(*nd_desc))
		return false;

	return true;
}

bool NeuroSystemManager::NetworkSave(bool bReload)
{
	if(!m_network)
		return false;

	return m_network->Save(bReload);
}

bool NeuroSystemManager::NetworkSaveAs(device::IODeviceFactory& nd_desc, neuro_u32 block_size, bool bReload)
{
	if(!m_network)
		return false;

	return m_network->SaveAs(nd_desc, block_size, bReload);
}

#include <functional>
#include "NeuroData/model/AbstractProducerModel.h"
network_ready_error::ReadyError* NeuroSystemManager::ReadyValidationCheck() const
{
	if (!m_network)
		return new network_ready_error::NetworkError;

	network_ready_error::_layer_error_vector layer_error_vector;

	std::function<void(const AbstractLayer* layer)> check_layer;
	check_layer = [&](const AbstractLayer* layer)->void
	{
		np::tensor::DataShape layer_ds = layer->GetOutTensorShape();
		if (layer_ds.GetDimSize() == 0)
		{
			layer_error_vector.push_back(network_ready_error::_LAYER_ERROR_INFO(layer, L"a size of layer's data shape is zero"));
			return;
		}

		const dp::model::AbstractProducerModel* binding_producer = NULL;

		const np::_neuro_binding_model_set& binding_set = layer->GetBindingSet();
		_neuro_binding_model_set::const_iterator it_binding = binding_set.begin();
		for (; it_binding != binding_set.end(); it_binding++)
		{
			NetworkBindingModel* binding = *it_binding;
			if (binding->GetBindingModelType() != _binding_model_type::data_producer)
				continue;

			binding_producer = (dp::model::AbstractProducerModel*)binding;
			if (!binding_producer->IsImageProcessingProducer())
			{
				np::tensor::DataShape producer_ds = binding_producer->GetDataShape();
				if (!producer_ds.IsEqual(layer_ds))
					layer_error_vector.push_back(network_ready_error::_LAYER_ERROR_INFO(layer, L"binding producer's data shape is not equal with layer's"));
			}
		}
		if (binding_producer == NULL)
		{
			layer_error_vector.push_back(network_ready_error::_LAYER_ERROR_INFO(layer, L"no binding producer"));
			return;
		}
	};

	const _input_layer_set& input_layer_set = m_network->GetInputLayerSet();
	for (_input_layer_set::const_iterator it = input_layer_set.begin(); it != input_layer_set.end(); it++)
		check_layer(*it);

	const _output_layer_set& output_layer_set = m_network->GetOutputLayerSet();
	for (_output_layer_set::const_iterator it = output_layer_set.begin(); it != output_layer_set.end(); it++)
		check_layer(*it);

	if (layer_error_vector.size() > 0)
	{
		network_ready_error::LayersError* ret = new network_ready_error::LayersError;
		ret->layer_error_vector = layer_error_vector;
		return ret;
	}
	return NULL;
}
