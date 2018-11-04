#if !defined(_TRAIN_FUNCTIONS_H)
#define _TRAIN_FUNCTIONS_H

#include "MiniBatchGenerator.h"
#include "backend/optimizer.h"

namespace np
{
	namespace engine
	{
		class TrainFunctions
		{
		public:
			TrainFunctions()
			{
				batch_gen = NULL;
				test_batch_gen = NULL;
				optimize_epoch = NULL;
			}
			~TrainFunctions()
			{
				delete batch_gen;
				delete test_batch_gen;
				delete optimize_epoch;
			}

			bool Initialize(core::math_device_type pdtype, core::cuda::CudaInstance* cuda_instance
				, const network::_LEARNING_INFO& info
				, const std::vector<neuro_float>& opt_parameters
				, dp::preprocessor::DataProvider& provider
				, dp::preprocessor::DataProvider* test_provider
				, const _producer_layer_data_vector& data_vector)
			{
				optimize_epoch = optimizer::OptimizeInEpoch::CreateInstance(pdtype, cuda_instance, info.optimizer_type, opt_parameters, info.optimizing_rule);
				if (optimize_epoch == NULL)
				{
					DEBUG_OUTPUT(L"failed create optimizer instance");
					return false;
				}

				switch ((network::_train_data_batch_type)info.data_batch_type)
				{
				case network::_train_data_batch_type::mini_batch_shuffle:
					batch_gen = new MiniBatchShuffleGenerator(provider, data_vector);
					break;
				default:
					batch_gen = new MiniBatchSequentialGenerator(provider, data_vector);
				}

				if (!batch_gen->Ready(provider.batch_size))
				{
					DEBUG_OUTPUT(L"failed ready of batch generator");
					return false;
				}

				if (test_provider)
				{
					test_batch_gen = new MiniBatchSequentialGenerator(*test_provider, data_vector);
					if (!test_batch_gen->Ready(provider.batch_size))
					{
						DEBUG_OUTPUT(L"failed ready of batch generator to test after learn");
						delete test_batch_gen;
						test_batch_gen = NULL;
					}
				}

				return true;
			}

			MiniBatchGenerator* batch_gen;
			MiniBatchSequentialGenerator* test_batch_gen;

			optimizer::OptimizeInEpoch* optimize_epoch;
		};
	}
}

#endif
