#include "stdafx.h"

#include "Simulator.h"

#include "NeuroKernel/engine/NeuralNetworkPredictor.h"
#include "NeuroKernel/engine/NeuralNetworkTrainer.h"

using namespace np;
using namespace np::simulate;

Simulator::Simulator(engine::NeuralNetworkEngine& network, neuro_u32 batch_size
	, dp::preprocessor::DataProvider* provider)
	: m_network(network), m_batch_size(batch_size)
{
	m_networkProcessor = NULL;
	m_provider = provider;
}

Simulator::~Simulator()
{
	delete m_provider;
	m_provider = NULL;

	delete m_networkProcessor;
	m_networkProcessor = NULL;
}

bool Simulator::ReadyToRun()
{
	if (!m_provider)
		return false;

	neuro_u32 batch_size = m_batch_size;
	if (batch_size == 0)
		batch_size = 1;
	if (batch_size > m_provider->GetDataCount())
		batch_size = m_provider->GetDataCount();

	m_networkProcessor = CreateProcessor();
	if (!m_networkProcessor)
		return false;

	if (!m_networkProcessor->Ready(batch_size))
	{
		DEBUG_OUTPUT(L"failed to Ready of Network. It might be when alloc buffers");
		return false;
	}

	return true;
}

LearnSimulator::LearnSimulator(engine::NeuralNetworkEngine& network
	, const engine::TRAIN_SETUP& setup
	, neuro_u32 batch_size)
	: Simulator(network, batch_size, setup.data.provider)
{
	memcpy(&m_train_setup, &setup, sizeof(engine::TRAIN_SETUP));
}

LearnSimulator::~LearnSimulator()
{
	if (m_train_setup.data.test_provider)
		delete m_train_setup.data.test_provider;
}

engine::NeuralNetworkProcessor* LearnSimulator::CreateProcessor()
{
	return new engine::NeuralNetworkTrainer(m_network);
}

bool LearnSimulator::Run()
{
	if (!m_provider)
		return false;

	return ((engine::NeuralNetworkTrainer*)m_networkProcessor)->Train(m_train_setup);
}

PredictSimulator::PredictSimulator(engine::NeuralNetworkEngine& network, neuro_u32 batch_size
	, const engine::PREDICT_SETUP& setup)
	: Simulator(network, batch_size, setup.provider)
{
	memcpy(&m_predict_setup, &setup, sizeof(engine::PREDICT_SETUP));
}

PredictSimulator::~PredictSimulator()
{
	if (m_predict_setup.result_writer)
		delete m_predict_setup.result_writer;
}

engine::NeuralNetworkProcessor* PredictSimulator::CreateProcessor()
{
	return new engine::NeuralNetworkPredictor(m_network);
}

bool PredictSimulator::Run()
{
	if (!m_provider)
		return false;

	return ((engine::NeuralNetworkPredictor*)m_networkProcessor)->Run(m_predict_setup);
}
