#if !defined(_NEURAL_NETWORK_ENTRY_SPEC_H)
#define _NEURAL_NETWORK_ENTRY_SPEC_H

#include "common.h"

const neuro_pointer32 layer_max = neuro_last32;
const neuro_pointer64 neuron_max = neuro_last64;

#pragma pack(push, 1)

namespace np
{
	namespace nsas
	{
		union _NEURAL_NETWORK_INFO
		{
			struct
			{
				neuro_u32 input_layer_count;
				neuro_u64 input_tensor_size;		// 입력 layer의 tensor 크기 총합. 약 42억 * 42억 미만
				neuro_u32 hidden_layer_count;		// 1 ~ 최대 42억 개의 layer
				neuro_u64 hidden_tensor_size;		// 히든 layer의 tensor 크기 총합. 약 42억 * 42억 미만

				neuro_u16 weight_init_type;		// 이게 필요한가???	. enum class _weight_init_type
				neuro_float weight_init_factor;

				neuro_u16 bias_init_type;		// 이게 필요한가???	. enum class _weight_init_type
				neuro_float bias_init_factor;
			};
			neuro_u8 reserved[128];
		};

		union _LEARNING_RATE_POLICY
		{
			struct
			{
				neuro_u16 type;	// enum class _lr_policy_type
				neuro_float lr_base;
				neuro_float gamma;
				neuro_u32 step;
				neuro_float power;
			};
			neuro_u8 reserved[32];
		};
		union _WEIGHT_NORM_POLICY
		{
			struct
			{
				neuro_u16 type;	// enum class _wn_policy_type
				neuro_float weight_decay;
			};
			neuro_u8 reserved[16];
		};

		union _OPTIMIZER_PARAMETER
		{
			struct
			{
				neuro_u16 count;
				neuro_float parameters[10];
			};
			neuro_u8 reserved[64];
		};

		union _LEARN_HISTORY
		{
			struct
			{
				neuron_error last_loss;
				neuro_float last_accuracy;
			};
			neuro_u8 reserved[32];
		};

		union _LEARN_INFO
		{
			struct
			{
				neuro_u16 optimizer_type;

				_LEARNING_RATE_POLICY lr_policy;	// 32 bytes
				_WEIGHT_NORM_POLICY wn_policy;		// 16 bytes

				neuro_u16 data_batch_type;
			};
			neuro_u8 reserved[128];
		};

		union _NEURO_ROOT_ENTRY
		{
			struct
			{
				_NEURAL_NETWORK_INFO network;	// 128 bytes
				_LEARN_INFO trainer;			// 128 bytes

				_OPTIMIZER_PARAMETER opt_params;	// 64 bytes
				_LEARN_HISTORY history;				// 32 bytes
			};

			neuro_u8 reserved[512];
		};

		struct _TENSOR_SHAPE
		{
			void Set(const tensor::TensorShape& src)
			{
				dense.SetVector(src);
				time = src.time_length;
			}
			void Get(tensor::TensorShape& src) const
			{
				dense.GetVector(src);
				src.time_length = time;
			}

			neuro_u32 time;
			_TYPED_VECTOR_N_DEF<neuro_u32, 8> dense;
		};

		union _INPUT_LAYER_ENTRY
		{
			struct
			{
				neuro_u32 uid;

				_TENSOR_SHAPE ts;	// 36 bytes.
			};// 42 bytes

			neuro_u8 reserved[64];
		};

		struct _FULLY_CONNECTED_LAYER_ENTRY
		{
			neuro_u32 output_count;		//	뉴런의 개수. 즉, 이 entry의 최종 출력 개수

			//각 입력(_INPUT_WEIGHT_ENTRY)의 weight_spec에 input X out 개의 weight이 있다.
			// bias_spec에 output_count 개의 bias weight이 있다.
		};	// 4 bytes

		struct _FILTER_ENTRY
		{
			neuro_16 kernel_width;
			neuro_16 kernel_height;

			neuro_16 stride_width;
			neuro_16 stride_height;

			neuro_16 pad_t, pad_b;
			neuro_16 pad_l, pad_r;
		};	// 16 bytes

		struct _CONVOLUTIONAL_LAYER_ENTRY
		{
			neuro_u32 channel_count;	// filter의 개수 즉, 출력 channel의 개수가 된다.

			neuro_u16 pad_type;	// 0 : define, 1 : valid, 2 : same(0으로 나머지를 패드로 채우는 것)

			_FILTER_ENTRY filter;	// 16 bytes

			neuro_16 dilation_width;
			neuro_16 dilation_height;

			// 각 입력(_INPUT_WEIGHT_ENTRY)의 weight_spec에 input_channel X kernel_width X kernel_height X out_channel 개의 weight이 있다.
			// bias_spec에 out_channel 개의 bias weight(64bit neuron_value)가 있다.
		};	// 27 bytes

		/*	입력 channel에 따라 출력 channel이 결정된다.
		즉, 이 entry는 바뀌지 않지만 출력 layer이 입력 layer에 따라 바뀔수 있다.
		*/
		struct _POOLING_LAYER_ENTRY
		{
			neuro_u16 type;	// _pooling_type. 0 : max pooling, 1 : ave pooling

			_FILTER_ENTRY filter;
		};	// 18 bytes

		struct _DROPOUT_LAYER_ENTRY
		{
			neuro_float dropout_rate;
		};	// 4 bytes

		struct _RECURRENT_LAYER_ENTRY
		{
			neuro_16 type;	// _rnn_type. 0 : lstm, 1 : gru

			neuro_u32 output_count;

			// for activity recognition
			neuro_u8 is_non_time_input;	// a non time varying data as input. so, transfer input data into each LSTM units.
			neuro_u8 is_avg_output;		// output has average one within each output of RNN unit

			neuro_u32 fix_time_length;	// if no input or is_non_time_input==1, it is enable to set time length
		};	// 11 bytes

		struct _CONCAT_LAYER_ENTRY
		{
			// 합치고자 하는 기준 index. 이 기준을 제외한 모든 차원의 크기가 같아야 한다.
			// 즉, 출력의 tensor shape는 axis에 따라 다르다.
			// axis에 batch는 포함되지 않는다.
			neuro_16 axis;	// 0 : all flatten, 1 : all flatten except T. 2 ~ : flatten under offset in dimension
		};	// 1 bytes

		struct _BATCH_NORMALIZATION_LAYER_ENTRY
		{
			neuro_float momentum;
			neuro_float eps;
		};	// 8 bytes

		struct _OUTPUT_LAYER_ENTRY
		{
			neuro_u16 loss_type;
		};

		struct _GROUP_LAYER_ENTRY
		{
			neuro_u32 child_first_layer_nid;
			neuro_u32 child_last_layer_nid;
		};	// 8 bytes

		union _LAYER_STRUCTURE_UNION // 128 bytes
		{
			_FULLY_CONNECTED_LAYER_ENTRY fc;
			_CONVOLUTIONAL_LAYER_ENTRY conv;
			_POOLING_LAYER_ENTRY pooling;
			_DROPOUT_LAYER_ENTRY dropout;
			_RECURRENT_LAYER_ENTRY rnn;
			_CONCAT_LAYER_ENTRY concat;
			_BATCH_NORMALIZATION_LAYER_ENTRY batch_norm;
			_OUTPUT_LAYER_ENTRY output;
			_GROUP_LAYER_ENTRY group;
			neuro_u8 entry_data[64];
		};

		struct _SLICE_INFO
		{
			tensor::TensorShape GetTensor(const tensor::TensorShape& input_tensor) const
			{
				tensor::TensorShape ret = input_tensor;

				// 1은 Time. 2부터 dimension
				if (slice_axis == 1)
					ret.time_length = slice_count;
				else if (slice_axis - 2 < ret.size())
					ret[slice_axis - 2] = slice_count;
				return ret;
			}

			neuro_u16 slice_axis;	// 0 : not using. 1 : T. 2 ~ : offset in dimension(like channelxheightxwidth or size)
			neuro_u32 slice_start;
			neuro_u32 slice_count;

			// if <0 : shift left. eg. -1 : ch(3) x h(4) x w(10) -> ch(3 x 4) x h(10)
			// if >0 : shift right. eg. 1 : ch(3) x h(4) x w(10) -> 1 x ch(3) x h(4 x 10)
			neuro_u16 channel_axis;	// 0 : default.
		};

		union _INPUT_ENTRY
		{
			struct 
			{
				neuro_u32 uid;
				_SLICE_INFO slice_info;
			};	// 14 bytes
			neuro_u8 entry_data[16];
		};

		struct _WEIGHT_INFO
		{
			neuro_u16 init_type;	// enum class _weight_init_type
			neuro_float init_scale;

			neuro_float mult_lr;// default : weight 1.0, bias는 2.0
			neuro_float decay;		// L1, L2 에 사용되는 Regularize decay 값. 보통 weight은 1, bias는 0
		};	// 16 bytes

		union _LAYER_BASIC_INFO
		{
			struct
			{
				neuro_u16 activation;	// enum class _activation_type

				_WEIGHT_INFO weight_info;	// 16 bytes
				_WEIGHT_INFO bias_info;
			};

			neuro_u8 reserved[64];
		}; // 64 bytes

		struct _LAYER_DATA_NIDS
		{
			neuro_u32 nid_count;
			neuro_u32 nid_vector[14];
		};	// 60 bytes

		union _LAYER_DATA_NID_SET
		{
			struct
			{
				neuro_u32 add_input_nid;	// 추가적인 _INPUT_ENTRY. concat, recurrent 에서는 layer로 결합. conv, fc 등에서는 각 input당 각 output으로 매치

				/*	각 layer type에 따른 n 개의 weight(neuron_value)이 있다.
				fc connected : output X input. output[input]
				convolutional : out_channel X in_channel X kernel_height X kernel_width
				recurrent : gate X hidden(output_count) X input
				*/
				// 각 layer type에 따른 n 개의 bias weight(neuron_value)이 있다. fc는 출력 개수만큼, conv는 filter(channel) 개수만큼
				// 추가 데이터의 nid. recurrent의 hidden gate의 weight, batch normalize에서 사용
				// 각 layer engine이 필요한 만큼 알아서 사용하도록 한다.
				_LAYER_DATA_NIDS data_nids;
			};	// 64

			neuro_u8 reserved[64];
		};

		struct _VIRTUAL_POSITION
		{
			/*	level_status
				0 : under previous layer
				1 : under previous layer and next layer is next level
				2 : next level
				3 : next level and next layer is next level
			*/
			neuro_u32 level_status;
			// not has input. but could have the input
			neuro_u32 virtual_input;
		};

		union _HIDDEN_LAYER_ENTRY
		{
			struct
			{
				neuro_u16 type;	// enum class _layer_type
				neuro_u32 uid;	// layer 고유 번호 최대 4,294,967,295 개 layer을 정의할 수 있다.

				_LAYER_BASIC_INFO basic_info;		// 64 bytes
				_LAYER_STRUCTURE_UNION function;	// 64 bytes

				_INPUT_ENTRY input;			// 16 bytes. 첫번째 입력.
				_INPUT_ENTRY side_input;	// 16 bytes. 대체로 rnn에서 사용하는 입력. connected previous rnn(same type) for cell states.

				_LAYER_DATA_NID_SET sub_nid_set;	// 64 bytes

				_VIRTUAL_POSITION virtual_position;	// using when no input
			};// 234 bytes

			neuro_u8 reserved[256];
		};	

		// grouping 하는 layer entry를 정의한다면???	그럼 여러개의 layer를 조합해서 새로운 기능의 layer를 개발할 수 있다.

	}
}
#pragma pack(pop)

#endif
