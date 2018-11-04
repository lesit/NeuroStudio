#pragma once

#include "NeuroKernel/engine/NeuralNetworkEngine.h"
#include "NeuroKernel/engine/NeuralNetworkTrainer.h"
#include "NeuroKernel/engine/NeuralNetworkPredictor.h"
#include "NeuroData/StreamWriter.h"

#include "../project/SimDefinition.h"

namespace np
{
	using namespace project;

	namespace simulate
	{
		enum class _sim_type{ train, predict };

		class Simulator
		{
		public:
			Simulator(engine::NeuralNetworkEngine& network, neuro_u32 batch_size
				, dp::preprocessor::DataProvider* provider);
			virtual ~Simulator();

			virtual bool ReadyToRun();
			virtual bool Run() = 0;

			const engine::NeuralNetworkEngine& GetNetwork() const{ return m_network; }

			virtual _sim_type GetType() const = 0;

		protected:
			const neuro_u32 m_batch_size;

			virtual engine::NeuralNetworkProcessor* CreateProcessor() = 0;

			engine::NeuralNetworkEngine& m_network;
			engine::NeuralNetworkProcessor* m_networkProcessor;

			dp::preprocessor::DataProvider* m_provider;
		};

		class LearnSimulator : public Simulator
		{
		public:
			LearnSimulator(engine::NeuralNetworkEngine& network
				, const engine::TRAIN_SETUP& setup
				, neuro_u32 batch_size);

			virtual ~LearnSimulator();

			bool Run() override;

			_sim_type GetType() const override{ return _sim_type::train; }

		protected:
			engine::NeuralNetworkProcessor* CreateProcessor() override;

			engine::TRAIN_SETUP m_train_setup;
		};

		class PredictSimulator : public Simulator
		{
		public:
			PredictSimulator(engine::NeuralNetworkEngine& network, neuro_u32 batch_size
				, const engine::PREDICT_SETUP& setup);

			virtual ~PredictSimulator();

			bool Run() override;

			_sim_type GetType() const override{ return _sim_type::predict; }

		protected:
			engine::NeuralNetworkProcessor* CreateProcessor() override;

			engine::PREDICT_SETUP m_predict_setup;
		};
	}
}
