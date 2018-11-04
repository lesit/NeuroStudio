#if !defined(_NP_TYPES_H)
#define _NP_TYPES_H

#include <vector>
#include <string>
#include <set>
#include <unordered_set>
#include <map>

namespace np
{
	typedef __int8	neuro_8;
	typedef __int16	neuro_16;
	typedef __int32	neuro_32;
	typedef __int64	neuro_64;

	typedef unsigned __int8		neuro_u8;
	typedef unsigned __int16	neuro_u16;
	typedef unsigned __int32	neuro_u32;
	typedef unsigned __int64	neuro_u64;

	typedef float	neuro_float;

	typedef neuro_u64 neuro_size_t;

	typedef neuro_u32 neuro_pointer32;
	typedef neuro_u64 neuro_pointer64;
	typedef neuro_u64 neuro_time;

	const neuro_u8 neuro_last8 = 0xff;
	const neuro_u16 neuro_last15 = 0x7fff;
	const neuro_u16 neuro_last16 = 0xffff;
	const neuro_pointer32 neuro_last31 = 0x7fffffff;
	const neuro_pointer32 neuro_last32 = 0xffffffff;
	const neuro_pointer32 neuro_last32_valid = 0xfffffffe;
	const neuro_pointer64 neuro_last63 = 0x7fffffffffffffff;
	const neuro_pointer64 neuro_last64 = 0xffffffffffffffff;

	const neuro_u32 invalid_uid32 = neuro_last32;
	const neuro_u64 invalid_uid64 = neuro_last64;

	/*	32bit�δ� ����� 1KB�϶� 4TB ���� ǥ�� ����. 4TB �̻��̸� ���ũ�⸦ 4KB/64KB ������ �ø��� �ȴ�.
		�׷�����.. ���߿� ��Ʈ��ũ���� ����ɰ� �����ϸ� 64bit�� �ϴ°� ������?.. �ƴ� ��¥�� ������ ���Ϳ��� ������״ϱ� 32bit�� ������..
		�ϴ� 64bit�� ����.
	*/
	typedef neuro_pointer64 neuro_block;
	const neuro_block neuro_lastblock=neuro_last64;
	const neuro_block neuro_emptyblock=0x0;

	typedef neuro_float neuron_value;
	typedef neuro_float neuron_error;

	typedef neuro_float	neuron_weight;

	typedef unsigned __int8 neuro_boolean;
	const neuro_boolean neuro_true=1;
	const neuro_boolean neuro_false=0;

	union _u64_union
	{
		struct
		{
			neuro_u32 lower;
			neuro_u32 upper;
		};
		neuro_u64 u64;
	};

	typedef std::vector<neuron_value> _std_value_vector;
	typedef std::vector<neuro_u32> _std_u32_vector;
	typedef std::set<neuro_u32> _u32_set;
	typedef std::unordered_set<neuro_u32> _unordered_u32_set;
	typedef std::vector<std::string> std_string_vector;
	typedef std::vector<std::wstring> std_wstring_vector;

	typedef std::vector<neuro_u64> _uid_vector;

	typedef std::vector<neuro_size_t> _neuro_size_t_vector;

	typedef std::vector<std::pair<neuron_value, neuron_value>> _scale_vector;

	enum _data_type{int32, int64, float32, float64, percentage, time, string, unknown};
	static const wchar_t* _data_type_string[]={L"32bit integer", L"64bit integer", L"32bit float", L"64bit float", L"percentage", L"time", L"string"};
	static _data_type ToDataType(const wchar_t* name)
	{
		for(neuro_u8 type=0;type<_countof(_data_type_string);type++)
			if(wcscmp(name, _data_type_string[type])==0)
				return (_data_type)type;
		
		return _data_type::unknown;;
	}

	typedef std::vector<_data_type> _data_type_vector;

	enum class _pad_type
	{
		user_define,	// user define
		valid,	// use valid pixels of input
		same,	// add zero-_padding around input so as to keep image size. ��, 0���� �е��� �ְڴٴ� ��
	};
	static const wchar_t* pad_type_string[] = { L"user define", L"valid", L"same" };
	static const wchar_t* ToString(_pad_type type)
	{
		if ((neuro_u32)type >= _countof(pad_type_string))
			return L"";
		return pad_type_string[(neuro_u32)type];
	}
}

#define BYTES_PER_KB	1024LL
#define BYTES_PER_MB	1048576LL
#define BYTES_PER_GB	1073741824LL
#define BYTES_PER_TB	1099511627776LL
#define BYTES_PER_PB	1125899906842624LL

#define MAX_NEURON			4225000000LL
#define MAX_NEURON_INPUT	65000

using namespace np;

#endif
