#include "stdafx.h"

#include "LossGraphWnd.h"

using namespace np::gui;
using namespace np::gui::win32;

CLossGraphWnd::CLossGraphWnd()
: CGraphWnd(m_source)
{

}

CLossGraphWnd::~CLossGraphWnd()
{

}

void CLossGraphWnd::AddData(neuro_float loss, neuro_float accuracy)
{
	m_source.AddData(loss, accuracy);

	ChangedSource();
}

void CLossGraphWnd::SetData(const std::vector<neuron_error>& loss_vector, const std::vector<neuro_float>& accuracy_vector)
{
	m_source.SetData(loss_vector, accuracy_vector);

	ChangedSource();
}

LossGraphSource::LossGraphSource()
{
	m_has_accuracy = false;
}

LossGraphSource::~LossGraphSource()
{
}

void LossGraphSource::SetData(const std::vector<neuro_float>& loss_vector, const std::vector<neuro_float>& accuracy_vector)
{
	m_loss_vector = loss_vector;
	m_accuracy_vector = accuracy_vector;
}

neuro_u32 LossGraphSource::GetTotalScrollDataCount() const
{
	return m_loss_vector.size();
}

bool LossGraphSource::IsValid(neuro_64 nStart, neuro_64 nCount)
{
	neuro_size_t end = min(m_loss_vector.size(), nStart + nCount);
	return nStart < end;
}

bool LossGraphSource::GetViewData(neuro_64 nStart, neuro_64 nCount, neuro_u32 max_ylabel, neuro_64 cur_data_pos, gui::_graph_frame& graphFrame)
{
	neuro_64 end = min(m_loss_vector.size(), nStart + nCount);
	if (nStart >= end)
		return false;

	graphFrame.graphViewVector.resize(m_has_accuracy ? 2 : 1);

	for (int graph_index = 0; graph_index < graphFrame.graphViewVector.size(); graph_index++)
	{
		_graph_view& graphView = graphFrame.graphViewVector[graph_index];

		graphView.heightRatio = 0.5;
		graphView.shapeType = gui::_graph_view::_shape_type::line;
		graphView.yLabelArray.clear();

		const bool is_loss_graph = graph_index == 0;
		const std::vector<neuro_float>& value_vector = is_loss_graph ? m_loss_vector : m_accuracy_vector;

		if (cur_data_pos >= end)
			cur_data_pos = end - 1;

		if (cur_data_pos >= nStart && cur_data_pos < end)
		{
			// 현재 마우스가 올려진 값
			if (is_loss_graph)
				graphView.curpos_yLabel_vector.push_back(_graphLabel(value_vector[cur_data_pos]));
			else
				graphView.curpos_yLabel_vector.push_back(_graphLabel(value_vector[cur_data_pos] * 100, util::StringUtil::Format(L"%.2f %%", value_vector[cur_data_pos] * 100).c_str()));

			graphFrame.has_cur_pos = true;
			graphFrame.curpos_xLabel = _graphLabel(cur_data_pos - nStart, util::StringUtil::Transform<wchar_t>(cur_data_pos).c_str());
		}
		else
			graphFrame.has_cur_pos = false;

		graphView.graphLineVector.resize(1);
		_graph_line& graph_line = graphView.graphLineVector[0];
		graph_line.clr = graph_index == 0 ? RGB(255, 0, 0) : RGB(0, 0, 255);

		_std_value_vector& graph_value_vector = graph_line.valueArray;
		graph_value_vector.resize(end - nStart);
		for (neuro_64 seq = 0; seq < end - nStart; seq++)
		{
			graph_value_vector[seq] = is_loss_graph ? value_vector[seq] : value_vector[seq]*100;

			if (seq % 50 == 0)
				graphFrame.xLabelVector.push_back(_graphLabel(seq, util::StringUtil::Transform<wchar_t>(nStart + seq).c_str()));
		}

		graphView.lower_boundary = *std::min_element(value_vector.begin(), value_vector.end());
		graphView.upper_boundary = *std::max_element(value_vector.begin(), value_vector.end());
		if (is_loss_graph)
		{
			graphView.yLabelArray.push_back(_graphLabel(graphView.lower_boundary));
			graphView.yLabelArray.push_back(_graphLabel(graphView.upper_boundary));
		}
		else
		{
			graphView.lower_boundary *= 100.f;
			graphView.upper_boundary *= 100.f;
			if (graphView.upper_boundary - graphView.lower_boundary > 60.f)
			{
				graphView.lower_boundary = 0.f;
				graphView.upper_boundary = 100.f;

				graphView.yLabelArray.push_back(_graphLabel(0.f, L"0 %"));
				if (max_ylabel > 2)
					graphView.yLabelArray.push_back(_graphLabel(50.f, L"50 %"));
				graphView.yLabelArray.push_back(_graphLabel(100.f, L"100 %"));
			}
			else
			{
				graphView.yLabelArray.push_back(_graphLabel(graphView.lower_boundary, util::StringUtil::Format(L"%.2f %%", graphView.lower_boundary).c_str()));
				graphView.yLabelArray.push_back(_graphLabel(graphView.upper_boundary, util::StringUtil::Format(L"%.2f %%", graphView.upper_boundary).c_str()));
			}
		}
	}
	return true;
}

CString LossGraphSource::GetDataTooltipLabel(neuro_u32 iGraph, neuro_u32 i) const
{
	return L"";
}
