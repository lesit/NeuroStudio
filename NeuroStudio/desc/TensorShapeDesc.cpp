#include "stdafx.h"

#include "TensorShapeDesc.h"

#include "util/StringUtil.h"

using namespace np::str_rc;

std::wstring TensorShapeDesc::GetDataShapeText(const np::tensor::DataShape& shape, bool bMultiLine)
{
	std::wstring ret;
	switch (shape.size())
	{
	case 1:
		ret = util::StringUtil::Transform<wchar_t>(shape[0]).c_str();
		break;
	case 2:
		ret = util::StringUtil::Transform<wchar_t>(shape[0]).c_str();
		ret += L" x ";
		ret += util::StringUtil::Transform<wchar_t>(shape[1]).c_str();
		break;
	case 3:
		ret = util::StringUtil::Transform<wchar_t>(shape[0]).c_str();
		if (bMultiLine)
			ret += L" channel\n\n";
		else
			ret += L" x ";

		ret += util::StringUtil::Transform<wchar_t>(shape[1]).c_str();
		ret += L" x ";
		ret += util::StringUtil::Transform<wchar_t>(shape[2]).c_str();
		break;
	default:
		ret = L"0";
	}
	return ret;
}

std::wstring TensorShapeDesc::GetTensorText(const np::tensor::TensorShape& ts, bool bMultiLine)
{
	if (ts.GetTensorSize() == 0)
		return L"0";

	std::wstring ret;
	if (ts.time_length > 1)
	{
		ret = util::StringUtil::Transform<wchar_t>(ts.time_length).c_str();
		if (bMultiLine)
			ret += L" time\r\n";
		else
			ret += L"t, ";
	}

	return ret + GetDataShapeText(ts, bMultiLine);
}
