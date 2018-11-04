#if !defined(_NEURAL_NETWORK_TYPES_H)
#define _NEURAL_NETWORK_TYPES_H

#include "np_types.h"

namespace np
{
	namespace network
	{
		enum class _layer_type
		{
			input = 0,
			output = 1,
			fully_connected = 100,	// _FULLY_CONNECTED_LAYER_ENTRY 가 되는 일반 뉴론 그룹
			convolutional = 101,	// _CONVOLUTIONAL_LAYER_ENTRY 가 되는 Convolutional 뉴론 그룹
			pooling = 102,			// _POOLING_LAYER_ENTRY 가 되는 Pooling 뉴론 그룹
			dropout = 103,			// _DROPOUT_LAYER_ENTRY 가 되는 Dropout 뉴론 그룹
			concat = 104,
			batch_norm = 105,		// batch norm
			rnn = 106,
			/*localnorm = 107,		// lrn, lcn*/
			unknown
		};

		static const wchar_t* ToString(_layer_type type)
		{
			switch (type)
			{
			case network::_layer_type::input:
				return L"Input";
			case network::_layer_type::output:
				return L"Output";
			case network::_layer_type::fully_connected:
				return L"Fully Connected";
			case network::_layer_type::convolutional:
				return L"Convolution";
			case network::_layer_type::pooling:
				return L"Pooling";
			case network::_layer_type::dropout:
				return L"Dropout";
			case network::_layer_type::concat:
				return L"Concatenation";
			case network::_layer_type::batch_norm:
				return L"Batch Normalization";
			case network::_layer_type::rnn:
				return L"Recurrent";
			}

			return L"";
		}

		enum class _pooling_type
		{
			max_pooling = 0,
			ave_pooling = 1,		// _POOLING_LAYER_ENTRY 가 되는 Average Pooling 뉴론 그룹
		};
		static const wchar_t* pooling_type_string[] = { L"Max", L"Average" };
		static const wchar_t* ToString(_pooling_type type)
		{
			if ((neuro_u32)type >= _countof(pooling_type_string))
				return L"";
			return pooling_type_string[(neuro_u32)type];
		}

		enum class _rnn_type
		{
			lstm = 0,
			gru = 1,
		};
		static const wchar_t* rnn_type_string[] = { L"lstm"/*, L"gru"*/ };
		static const wchar_t* ToString(_rnn_type type)
		{
			if ((neuro_u32)type >= _countof(rnn_type_string))
				return L"";
			return rnn_type_string[(neuro_u32)type];
		}

		enum class _layer_data_type { weight, bias, other };
		struct _LAYER_DATA_INFO
		{
			_layer_data_type wtype;
			neuro_u32 size;
		};
		typedef std::vector<_LAYER_DATA_INFO> _layer_data_info_vector;

		enum class _weight_init_type
		{
			Zero,
			Constant,
			Gaussian,
			He,
			LeCun,
			Xavier,
		};
		static const wchar_t* weight_init_type_string[] = { L"Zero", L"Constant", L"Gaussian", L"He", L"LeCun", L"Xavier" };
		static const wchar_t* ToString(_weight_init_type type)
		{
			if ((neuro_u32)type >= _countof(weight_init_type_string))
				return L"";
			return weight_init_type_string[(neuro_u32)type];
		}

		struct _LAYER_WEIGHT_INFO
		{
			_weight_init_type init_type;	
			neuro_float init_scale;

			neuro_float mult_lr;// default : weight 1.0, bias는 2.0
			neuro_float decay;		// L1, L2 에 사용되는 Regularize decay 값. 보통 weight은 1, bias는 0
		};	

		enum class _activation_type
		{
			none = 0,
			sigmoid = 1,
			tahn = 2,
			reLU = 3,
			leakyReLU = 4,
			eLU = 5,
			softmax = 6,
			unknown
		};
		static const wchar_t* activation_type_string[] = {L"none", L"sigmoid", L"tahn", L"reLU", L"leaky-reLU", L"eLU", L"softmax" };
		static const wchar_t* ToString(_activation_type type)
		{
			if ((neuro_u32)type >= _countof(activation_type_string))
				return L"";
			return activation_type_string[(neuro_u32)type];
		}

		enum class _loss_type
		{
			MSE,					// regression 에서 사용
			CrossEntropy,			// binary classification에서 사용
			CrossEntropyMulticlass,	// multi classification에서 사용
			SigmoidCrossEntropy,
			SoftmaxWithLoss,
		};
		static const wchar_t* loss_type_string[] = { L"MSE", L"CrossEntropy", L"CrossEntropyMulticlass", L"SigmoidCrossEntropy", L"SoftmaxWithLoss"};
		static const wchar_t* ToString(_loss_type type)
		{
			if ((neuro_u32)type >= _countof(loss_type_string))
				return L"";
			return loss_type_string[(neuro_u32)type];
		}

		enum class _optimizer_type
		{
			SGD,
			Adagrad,
			Adam,
			RMSprop,
		};
		static const wchar_t* optimizer_type_string[] = { L"SGD", L"Adagrad", L"Adam", L"RMSprop" };
		static const wchar_t* ToString(_optimizer_type type)
		{
			if ((neuro_u32)type >= _countof(optimizer_type_string))
				return L"";
			return optimizer_type_string[(neuro_u32)type];
		}

		enum class _lr_policy_type
		{
			Fix,
			StepByBatch,
			StepByEpoch,
			Random,
		};
		static const wchar_t* lr_policy_type_string[] = { L"Fix", L"Step by Batch", L"Step by Epoch", L"Random" };
		static const wchar_t* ToString(_lr_policy_type type)
		{
			if ((neuro_u32)type >= _countof(lr_policy_type_string))
				return L"";
			return lr_policy_type_string[(neuro_u32)type];
		}

		struct _LEARNING_RATE_POLICY
		{
			_lr_policy_type type;
			neuro_float lr_base;
			neuro_float gamma;
			neuro_u32 step;
			neuro_float power;
		};

		enum class _wn_policy_type	// weight normalize policy type
		{
			none,
			L1,
			L2,
			L12
		};
		static const wchar_t* wn_policy_type_string[] = { L"none", L"L1", L"L2", L"L1 & L2" };
		static const wchar_t* ToString(_wn_policy_type type)
		{
			if ((neuro_u32)type >= _countof(wn_policy_type_string))
				return L"";
			return wn_policy_type_string[(neuro_u32)type];
		}

		struct _WEIGHT_NORM_POLICY
		{
			_wn_policy_type type;
			neuro_float weight_decay;
		};

		struct _OPTIMIZING_RULE
		{
			_LEARNING_RATE_POLICY lr_policy;
			_WEIGHT_NORM_POLICY wn_policy;
		};

		enum class _train_data_batch_type
		{
			mini_batch_sequential,
			mini_batch_shuffle,
		};
		static const wchar_t* train_data_batch_type_string[] = {
			L"mini_batch_sequential",
			L"mini_batch_shuffle",
		};

		static const wchar_t* ToString(_train_data_batch_type type)
		{
			if ((neuro_u32)type >= _countof(train_data_batch_type_string))
				return L"";
			return train_data_batch_type_string[(neuro_u32)type];
		}

		struct _LEARNING_INFO
		{
			_optimizer_type optimizer_type;
			_OPTIMIZING_RULE optimizing_rule;

			_train_data_batch_type data_batch_type;	
		};
	}
}

#endif
