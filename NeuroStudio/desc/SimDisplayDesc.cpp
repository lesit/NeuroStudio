#include "stdafx.h"
#include "SimDisplayDesc.h"

using namespace np::str_rc;

const wchar_t* SimDisplayDesc::GetViewShapeString(windows::_graph_view::_shape_type type)
{
	if (type == windows::_graph_view::_shape_bar)
		return L"bar";
	else if (type == windows::_graph_view::_shape_line)
		return L"line";
	else
		return L"dot";
}

windows::_graph_view::_shape_type SimDisplayDesc::GetViewShapeType(const wchar_t* strType)
{
	if (wcscmp(strType, GetViewShapeString(windows::_graph_view::_shape_bar))==0)
		return windows::_graph_view::_shape_bar;
	else if (wcscmp(strType, GetViewShapeString(windows::_graph_view::_shape_line)) == 0)
		return windows::_graph_view::_shape_line;
	return windows::_graph_view::_shape_dot;
}
