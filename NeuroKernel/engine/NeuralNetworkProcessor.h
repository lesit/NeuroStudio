#pragma once

#include "NeuralNetworkEngine.h"

#include "NeuroData/reader/DataProvider.h"
#include "layers/OutputLayerEngine.h"

#include <unordered_map>

namespace np
{
	namespace engine
	{
		// <layer uid, producer uid>
		typedef std::unordered_map<neuro_u32, neuro_u32> _layer_producer_uid_map;

		struct _analysis_onehot
		{
			neuro_u32 out_accumulate;
			neuro_u32 target_accumulate;
			neuro_u32 accord_accumulate;
		};

		struct _output_analysys_onehot
		{
			void operator = (const _output_analysys_onehot& src)
			{
				layer = src.layer;
				results = src.results;
				test_after_learn = src.test_after_learn;
			}
			const AbstractLayerEngine* layer;
			std::vector<_analysis_onehot> results;
			std::vector<_analysis_onehot> test_after_learn;
		};
		typedef std::unordered_map<neuro_u32, _output_analysys_onehot*> _output_analysys_onehot_map;
		typedef std::vector<_output_analysys_onehot> _output_analysis_onehot_vector;

		namespace analysis_loss
		{
			/*
			struct _layer_accuracy_info
			{
				neuro_u32 layer_id;
				neuro_size_t accord_count;
			};
			typedef std::vector<_layer_accuracy_info> _layer_accuracy_info_vector;
			struct _batch_info
			{
				_batch_info()
				{
					loss = 0;
					total_argmax_accord_count = 0;
					accuracy = 0;
				}
				~_batch_info() {}

				_batch_info& operator = (const _batch_info& src)
				{
					loss = src.loss;
					layer_accuracy_info_vector = src.layer_accuracy_info_vector;
					total_argmax_accord_count = src.total_argmax_accord_count;
					accuracy = src.accuracy;
					return *this;
				}
				neuron_error loss;

				_layer_accuracy_info_vector layer_accuracy_info_vector;
				neuro_size_t total_argmax_accord_count;
				neuro_float accuracy;
			};
			typedef std::vector<_batch_info> _batch_vector;
			*/
			struct _epoch_info
			{
				_epoch_info()
				{
					reset();
				}

				void reset()
				{
					data_count = 0;

					loss = -1;
					argmax_accord = 0;
					accuracy = 0;
				}

				neuro_size_t data_count;

				neuron_error loss;
				neuro_size_t argmax_accord;
				neuro_float accuracy;
			};
		}

		struct _INPUT_DATA_INFO
		{
			neuro_size_t batch_size;
			neuro_size_t data_count;
			neuro_size_t batch_count;
		};

		struct LEARN_EPOCH_ELAPSE
		{
			neuro_float batch_gen;
			neuro_float forward;
			neuro_float learn;
		};

		enum class _signal_type { learn_start, learn_end, learn_epoch_start, learn_epoch_end, predict_start, predict_end, batch_start, batch_end };
		struct _NET_SIGNAL_INFO
		{
			virtual ~_NET_SIGNAL_INFO() {}
			virtual _signal_type GetType() const = 0;
			virtual _NET_SIGNAL_INFO* Clone() const = 0;

			void operator = (const _NET_SIGNAL_INFO& src)
			{
				total_elapse = src.total_elapse;
			}

			neuro_float total_elapse;
		};

		enum class _sigout { sig_continue, sig_stop, sig_epoch_stop };
		class RecvSignal
		{
		public:
			virtual ~RecvSignal() {}
			virtual _sigout network_signal(const _NET_SIGNAL_INFO& info, std::unordered_set<neuro_u32>* epoch_start_onehot_output = NULL) = 0;
		};

		struct _LEARN_START_INFO : public _NET_SIGNAL_INFO
		{
			_signal_type GetType() const override { return _signal_type::learn_start; }
			_NET_SIGNAL_INFO* Clone() const override
			{
				_LEARN_START_INFO* ret = new _LEARN_START_INFO;
				*ret = *this;
				return ret;
			}

			void operator = (const _LEARN_START_INFO& src)
			{
				__super::operator = (src);

				learn_data = src.learn_data;
				has_test = src.has_test;
				test_data = src.test_data;
			}
			_INPUT_DATA_INFO learn_data;
			bool has_test;
			_INPUT_DATA_INFO test_data;
		};
		struct _LEARN_END_INFO : public _NET_SIGNAL_INFO
		{
			_signal_type GetType() const override { return _signal_type::learn_end; }
			_NET_SIGNAL_INFO* Clone() const override
			{
				_LEARN_END_INFO* ret = new _LEARN_END_INFO;
				*ret = *this;
				return ret;
			}

			void operator = (const _LEARN_END_INFO& src)
			{
				__super::operator = (src);
				epoch = src.epoch;
			}

			neuro_size_t epoch;
		};

		struct _EPOCH_INFO : public _NET_SIGNAL_INFO
		{
			void operator = (const _EPOCH_INFO& src)
			{
				__super::operator = (src);
				epoch = src.epoch;
			}
			neuro_size_t epoch;
		};

		struct _EPOCH_START_INFO : public _EPOCH_INFO
		{
			_signal_type GetType() const override { return _signal_type::learn_epoch_start; }
			_NET_SIGNAL_INFO* Clone() const override
			{
				_EPOCH_START_INFO* ret = new _EPOCH_START_INFO;
				*ret = *this;
				return ret;
			}

			void operator = (const _EPOCH_START_INFO& src)
			{
				__super::operator = (src);
				learn_rate = src.learn_rate;
			}
			neuro_float learn_rate;
		};
		struct _EPOCH_END_INFO : public _EPOCH_INFO
		{
			_signal_type GetType() const override { return _signal_type::learn_epoch_end; }
			_NET_SIGNAL_INFO* Clone() const override
			{
				_EPOCH_END_INFO* ret = new _EPOCH_END_INFO;
				*ret = *this;
				return ret;
			}

			void operator = (const _EPOCH_END_INFO& src)
			{
				__super::operator = (src);

				analysis = src.analysis;
				has_test = src.has_test;
				test_after_learn_analysis = src.test_after_learn_analysis;

				output_onehot_result_vector = src.output_onehot_result_vector;

				epoch_elapse = src.epoch_elapse;
				elapse = src.elapse;
			}
			analysis_loss::_epoch_info analysis;
			bool has_test;
			analysis_loss::_epoch_info test_after_learn_analysis;

			_output_analysis_onehot_vector output_onehot_result_vector;

			neuro_float epoch_elapse;
			LEARN_EPOCH_ELAPSE elapse;
		};

		struct _PREDICT_START_INFO : public _NET_SIGNAL_INFO
		{
			_signal_type GetType() const override { return _signal_type::predict_start; }
			_NET_SIGNAL_INFO* Clone() const override
			{
				_PREDICT_START_INFO* ret = new _PREDICT_START_INFO;
				*ret = *this;
				return ret;
			}

			void operator = (const _PREDICT_START_INFO& src)
			{
				__super::operator = (src);
				data = src.data;
			}
			_INPUT_DATA_INFO data;
		};

		struct _PREDICT_END_INFO : public _NET_SIGNAL_INFO
		{
			_signal_type GetType() const override { return _signal_type::predict_end; }
			_NET_SIGNAL_INFO* Clone() const override
			{
				_PREDICT_END_INFO* ret = new _PREDICT_END_INFO;
				*ret = *this;
				return ret;
			}

			void operator = (const _PREDICT_END_INFO& src)
			{
				__super::operator = (src);
				batch_gen_elapse = src.batch_gen_elapse;
				forward_elapse = src.forward_elapse;
			}

			neuro_float batch_gen_elapse;
			neuro_float forward_elapse;
		};

		struct BATCH_STATUS_INFO : public _NET_SIGNAL_INFO
		{
			_signal_type GetType() const override { return type; }
			_NET_SIGNAL_INFO* Clone() const override
			{
				BATCH_STATUS_INFO* ret = new BATCH_STATUS_INFO;
				*ret = *this;
				return ret;
			}

			void operator = (const BATCH_STATUS_INFO& src)
			{
				__super::operator = (src);
				type = src.type;
				batch_size = src.batch_size;
				batch_no = src.batch_no;
				learn_rate = src.learn_rate;
				loss = src.loss;
			}
			_signal_type type;

			neuro_u32 batch_size;
			neuro_u32 batch_no;
			neuro_float learn_rate;

			neuro_float loss;
		};

		struct TRAIN_ANALYZE
		{
			bool bAnalyzeArgmaxAccuracy;
			bool bAnalyzeLossHistory;
		};

		using namespace dp::preprocessor;
		class NeuralNetworkProcessor
		{
		public:
			NeuralNetworkProcessor(NeuralNetworkEngine& engine);
			virtual ~NeuralNetworkProcessor();

			virtual bool IsLearn() const = 0;

			virtual bool Ready(neuro_u32 batch_size = 1);

		protected:
			static AbstractProducer* NeuralNetworkProcessor::FindLayerBindingProducer(DataProvider& provider, const AbstractLayerEngine& engine);

			bool Propagate(bool bTrain, neuro_u32 batch_size);

			NeuralNetworkEngine& m_network;

			const _uid_engine_map& m_uid_layer_map;

			const _hidden_engine_vector& m_engine_vector;
		};
	}
}
