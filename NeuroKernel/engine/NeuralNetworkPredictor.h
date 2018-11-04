#pragma once

#include "NeuralNetworkProcessor.h"

#include "NeuroData/StreamWriter.h"

namespace np
{
	namespace engine
	{
		struct PREDICT_SETUP
		{
			dp::preprocessor::DataProvider* provider;

			dp::StreamWriter* result_writer;

			RecvSignal* recv_signal;
		};

		class NeuralNetworkPredictor : public NeuralNetworkProcessor
		{
		public:
			NeuralNetworkPredictor(NeuralNetworkEngine& nn);
			~NeuralNetworkPredictor();

			bool IsLearn() const override { return false; }

//			bool Run(const _VALUE_VECTOR& input, const _VALUE_VECTOR* output = NULL);
			bool Run(const engine::PREDICT_SETUP& setup);
		};
	}
}
