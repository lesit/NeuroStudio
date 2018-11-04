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
				neuro_u64 input_tensor_size;		// �Է� layer�� tensor ũ�� ����. �� 42�� * 42�� �̸�
				neuro_u32 hidden_layer_count;		// 1 ~ �ִ� 42�� ���� layer
				neuro_u64 hidden_tensor_size;		// ���� layer�� tensor ũ�� ����. �� 42�� * 42�� �̸�

				neuro_u16 weight_init_type;		// �̰� �ʿ��Ѱ�???	. enum class _weight_init_type
				neuro_float weight_init_factor;

				neuro_u16 bias_init_type;		// �̰� �ʿ��Ѱ�???	. enum class _weight_init_type
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
			neuro_u32 output_count;		//	������ ����. ��, �� entry�� ���� ��� ����

			//�� �Է�(_INPUT_WEIGHT_ENTRY)�� weight_spec�� input X out ���� weight�� �ִ�.
			// bias_spec�� output_count ���� bias weight�� �ִ�.
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
			neuro_u32 channel_count;	// filter�� ���� ��, ��� channel�� ������ �ȴ�.

			neuro_u16 pad_type;	// 0 : define, 1 : valid, 2 : same(0���� �������� �е�� ä��� ��)

			_FILTER_ENTRY filter;	// 16 bytes

			neuro_16 dilation_width;
			neuro_16 dilation_height;

			// �� �Է�(_INPUT_WEIGHT_ENTRY)�� weight_spec�� input_channel X kernel_width X kernel_height X out_channel ���� weight�� �ִ�.
			// bias_spec�� out_channel ���� bias weight(64bit neuron_value)�� �ִ�.
		};	// 27 bytes

		/*	�Է� channel�� ���� ��� channel�� �����ȴ�.
		��, �� entry�� �ٲ��� ������ ��� layer�� �Է� layer�� ���� �ٲ�� �ִ�.
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
			// ��ġ���� �ϴ� ���� index. �� ������ ������ ��� ������ ũ�Ⱑ ���ƾ� �Ѵ�.
			// ��, ����� tensor shape�� axis�� ���� �ٸ���.
			// axis�� batch�� ���Ե��� �ʴ´�.
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

				// 1�� Time. 2���� dimension
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

			neuro_float mult_lr;// default : weight 1.0, bias�� 2.0
			neuro_float decay;		// L1, L2 �� ���Ǵ� Regularize decay ��. ���� weight�� 1, bias�� 0
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
				neuro_u32 add_input_nid;	// �߰����� _INPUT_ENTRY. concat, recurrent ������ layer�� ����. conv, fc ����� �� input�� �� output���� ��ġ

				/*	�� layer type�� ���� n ���� weight(neuron_value)�� �ִ�.
				fc connected : output X input. output[input]
				convolutional : out_channel X in_channel X kernel_height X kernel_width
				recurrent : gate X hidden(output_count) X input
				*/
				// �� layer type�� ���� n ���� bias weight(neuron_value)�� �ִ�. fc�� ��� ������ŭ, conv�� filter(channel) ������ŭ
				// �߰� �������� nid. recurrent�� hidden gate�� weight, batch normalize���� ���
				// �� layer engine�� �ʿ��� ��ŭ �˾Ƽ� ����ϵ��� �Ѵ�.
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
				neuro_u32 uid;	// layer ���� ��ȣ �ִ� 4,294,967,295 �� layer�� ������ �� �ִ�.

				_LAYER_BASIC_INFO basic_info;		// 64 bytes
				_LAYER_STRUCTURE_UNION function;	// 64 bytes

				_INPUT_ENTRY input;			// 16 bytes. ù��° �Է�.
				_INPUT_ENTRY side_input;	// 16 bytes. ��ü�� rnn���� ����ϴ� �Է�. connected previous rnn(same type) for cell states.

				_LAYER_DATA_NID_SET sub_nid_set;	// 64 bytes

				_VIRTUAL_POSITION virtual_position;	// using when no input
			};// 234 bytes

			neuro_u8 reserved[256];
		};	

		// grouping �ϴ� layer entry�� �����Ѵٸ�???	�׷� �������� layer�� �����ؼ� ���ο� ����� layer�� ������ �� �ִ�.

	}
}
#pragma pack(pop)

#endif
