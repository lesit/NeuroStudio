#include "NumericProducerModel.h"

using namespace np;
using namespace np::dp;
using namespace np::dp::model;

NumericProducerModel::NumericProducerModel(DataProviderModel& provider, neuro_u32 uid)
: AbstractProducerModel(provider, uid)
{
	m_onehot_encoding_size = 0;
}

NumericProducerModel::~NumericProducerModel()
{
}

_NUMERIC_USING_SOURCE_COLUMNS NumericProducerModel::GetUsingSourceColumnVector() const
{
	_NUMERIC_USING_SOURCE_COLUMNS ret;
	ret.max_ma = 1;
	ret.ma_count = 0;
	if (GetInput() == NULL)
		return ret;

	neuro_u32 in_column_count = GetInput()->GetColumnCount();

	std::map<neuro_u32, neuro_u32>::const_iterator it = m_using_colum_map.begin();
	for (; it != m_using_colum_map.end(); it++)
	{
		if (it->first >= in_column_count)
			continue;

		if (ret.max_ma < it->second)
			ret.max_ma = it->second;

		if (it->second > 1)
			++ret.ma_count;

		ret.column_vector.push_back(_NUMERIC_SOURCE_COLUMN(it->first, it->second));
	}
	return ret;
}

std::string NumericProducerModel::MakeNdfPath(const std::string& source_name) const
{
	if (GetInput() == NULL || GetInput()->GetModelType() != _model_type::reader)
		return "";

	if (m_using_colum_map.size()==0)
		return "";

	std::string ndf_name = source_name;
	if (ndf_name.empty())
		return "";

	ndf_name.append(".nf");
	if (m_onehot_encoding_size > 1)
		ndf_name.append("_h").append(util::StringUtil::Transform<char>(m_onehot_encoding_size));
	else if (GetMinScale() == 0)
		ndf_name.append("_0");

	_NUMERIC_USING_SOURCE_COLUMNS columns = GetUsingSourceColumnVector();
	for (neuro_u32 i=0;i<columns.column_vector.size();i++)
	{
		ndf_name.append("_").append(util::StringUtil::Format<char>("%u", columns.column_vector[i].index));
		if (columns.column_vector[i].ma>1)
			ndf_name.append(util::StringUtil::Format("_ma[%u]", columns.column_vector[i].ma));
	}
	ndf_name.append(".ndf");
	return ndf_name;
}

_ndf_dim_type NumericProducerModel::GetNdfDimType() const
{
	return _ndf_dim_type::all_fix;
}

tensor::DataShape NumericProducerModel::GetDataShape() const
{
	tensor::DataShape ret({ 0 });
	if (GetInput() == NULL)
		return ret;

	if (m_using_colum_map.size() == 0)
		return tensor::DataShape({ 0 });

	if (m_onehot_encoding_size>1)
		return tensor::DataShape({ 1 });

	neuro_u32 in_column_count = GetInput()->GetColumnCount();

	neuro_u32 using_count = 0;
	std::map<neuro_u32, neuro_u32>::const_iterator it = m_using_colum_map.begin();
	for (; it != m_using_colum_map.end(); it++)
	{
		if (it->first >= in_column_count)
			continue;

		++using_count;
	}

	return tensor::DataShape({ using_count });
}

neuro_u32 NumericProducerModel::GetAvailableStartPosition() const
{
	if (GetInput() == NULL)
		return 0;

	neuro_u32 in_column_count = GetInput()->GetColumnCount();

	neuro_u32 max_ma = 1;

	std::map<neuro_u32, neuro_u32>::const_iterator it = m_using_colum_map.begin();
	for (; it != m_using_colum_map.end(); it++)
	{
		if (it->first >= in_column_count)
			continue;

		if (it->second > 1)
		{
			if (max_ma < it->second)
				max_ma = it->second;
		}
	}
	return max_ma - 1;
}
