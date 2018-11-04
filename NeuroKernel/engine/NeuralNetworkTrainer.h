#if !defined(_NEURAL_NETWORK_TRAINER_H)
#define _NEURAL_NETWORK_TRAINER_H

#include "NeuralNetworkProcessor.h"
#include "util/LogFileWriter.h"

namespace np
{
	namespace engine
	{
		enum class _learn_type{ learn_test_both, learn, test };
		struct LEARNNING_SETUP
		{
			_learn_type learn_type;
			bool isWeightInit;

			neuro_size_t epoch_count;
			TRAIN_ANALYZE analyze;
		};

		struct LEARN_DATA_INFO
		{
			dp::preprocessor::DataProvider* provider;
			dp::preprocessor::DataProvider* test_provider;
		};

		struct TRAIN_SETUP
		{
			LEARNNING_SETUP learn;

			LEARN_DATA_INFO data;

			util::LogFileWriter* log_writer;

			RecvSignal* recv_signal;
		};

		namespace loss
		{
			class LossFunction;
		}

		namespace optimizer
		{
			class OptimizeInEpoch;
		}
		class MiniBatchGenerator;
		class NeuralNetworkTrainer : public NeuralNetworkProcessor
		{
		public:
			NeuralNetworkTrainer(NeuralNetworkEngine& nn);
			virtual ~NeuralNetworkTrainer();

			bool IsLearn() const override { return true; }

			bool Ready(neuro_u32 batch_size) override;
			bool Train(const TRAIN_SETUP& setup);

		protected:
			bool RunEpoch(np::Timer& total_timer, bool isTrainLearn, optimizer::OptimizeInEpoch& optimize_epoch
				, MiniBatchGenerator& batch_gen
				, neuro_u32 epoch
				, analysis_loss::_epoch_info& analysis_epoch_info
				, LEARN_EPOCH_ELAPSE* elapse
				, _output_analysys_onehot_map& output_analysys_onehot_map
				, bool is_test_output_analysis
				, RecvSignal* recv_batch_signal, _sigout* sig_ret);

			void MakeEpochOnehotResult(const _output_analysis_onehot_vector& output_onehot_result_vector, std::wstring& result_status);

			neuron_error SumOutputLayerLoss();
			bool Learn(const neuro_u32 batch_size);

		private:
			std::unordered_map<neuro_u32, tensor::_NEURO_TENSOR_DATA> m_target_buffer_map;	// target은 무조건 network의 pdtype에 맞춘다.

			_NEURO_TENSOR_DATA m_cpu_out_buffer;
			_TYPED_TENSOR_DATA<void*,4> m_cpu_target_buffer;

			class AnalysisLoss
			{
			public:
				AnalysisLoss()
				{
					condition.bAnalyzeArgmaxAccuracy = true;
					condition.bAnalyzeLossHistory = true;
				}
				engine::TRAIN_ANALYZE condition;
			};
			AnalysisLoss *m_analyze_loss;
		};
	}
}
#endif
