#include "stdafx.h"

#include "ConcatLayerConfigure.h"

using namespace np;
using namespace np::network;

_CONCAT_INFO ConcatLayerConfigure::GetConcatInfo(const HiddenLayer& layer)
{
	const network::_slice_input_vector& input_vector = layer.GetInputVector();

	_CONCAT_INFO ret;
	if (input_vector.size() == 0)
		return ret;

	std::vector<tensor::TensorShape> input_ts_vector;
	input_ts_vector.resize(input_vector.size());
	for (neuro_size_t i = 0; i < input_vector.size(); i++)
	{
		input_ts_vector[i] = input_vector[i].GetTensor();

		tensor::TensorShape& ts = input_ts_vector[i];
		int last = ts.size() - 1;
		if (last < 0)
			continue;

		for (; last >= 0; last--)
		{
			if (ts[last] > 1)
				break;
		}
		++last;
		ts.erase(ts.begin() + last, ts.end());

		// 단 한개의 dimension을 가지고 1인 경우 data가 없는게 아니므로, 최초 1은 유지해준다.
		if (ts.size() == 0 && input_vector[i].GetTensor()[0]==1)
			ts.push_back(1);
	}

	ret.concat_axis = input_ts_vector[0].size() - 1;
	if (ret.concat_axis >= 0)
	{
		for (int i = 1; i < input_ts_vector.size(); i++)
		{
			ret.concat_axis = input_ts_vector[0].FindMatchAxisCount(input_ts_vector[i], ret.concat_axis + 1);
			if (ret.concat_axis < 0)	// time도 다르면 더이상 비교할 필요가 없다.
				break;
		}
	}
	// concat_axis = -1 : time까지도 다른 것이다.
	// concat_axis = 0 : time까지는 같다.

	ret.join_ts = input_ts_vector[0];
	ret.join_ts.resize(max(ret.concat_axis - 1, 0));

	// concat_axis 다음부터 일치성 검사를 해야한다.
	int first_axes_count = input_ts_vector[0].size();
	neuro_size_t i = 1;
	for (; i < input_ts_vector.size(); i++)
	{
		const tensor::TensorShape& in_ts = input_ts_vector[i];
		if (in_ts.size() != first_axes_count)
			break;

		int axis = ret.concat_axis + 1;
		for (; axis < first_axes_count; axis++)
		{
			if (in_ts[axis] != input_ts_vector[0][axis])
				break;
		}
		if (axis < first_axes_count)
			break;
	}
	if (i < input_ts_vector.size())	// axis 이후에 불일치한다. 따라서, axis 부터의 dim size를 더해줘야 한다
	{
		for (i=0; i < input_ts_vector.size(); i++)
		{
			const tensor::TensorShape& in_ts = input_ts_vector[i];
			ret.concat_axis_size += ret.concat_axis < 0 ? in_ts.GetTensorSize() : in_ts.GetDimSize(ret.concat_axis);
		}
	}
	else// axis만 빼고 모두 일치한다!
	{	
		// 모두다 일치할 경우
		if (ret.concat_axis == first_axes_count)
			--ret.concat_axis;

		for (i = 0; i < input_ts_vector.size(); i++)
		{
			const tensor::TensorShape& in_ts = input_ts_vector[i];
			ret.concat_axis_size += ret.concat_axis < 0 ? in_ts.time_length : in_ts[ret.concat_axis];
		}
		ret.concat_ts.assign(input_ts_vector[0].begin() + (ret.concat_axis + 1), input_ts_vector[0].end());
	}
	return ret;
}

tensor::TensorShape ConcatLayerConfigure::MakeOutTensorShape(const HiddenLayer& layer) const
{
	_CONCAT_INFO concat_info = GetConcatInfo(layer);
	return concat_info.toTensor();
}
