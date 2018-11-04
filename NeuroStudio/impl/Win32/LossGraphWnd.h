#pragma once

#include "gui/win32/GraphWnd.h"

class LossGraphSource : public gui::GraphDataSourceAbstract
{
public:
	LossGraphSource();
	virtual ~LossGraphSource();

	void SetHasAccuracy(bool has){ m_has_accuracy=has; }
	
	inline void AddData(neuro_float loss, neuro_float accuracy=0.f)
	{
		m_loss_vector.push_back(loss);
		if (m_has_accuracy)
			m_accuracy_vector.push_back(accuracy);
	}
	void SetData(const std::vector<neuro_float>& loss_vector, const std::vector<neuro_float>& accuracy_vector);

	void Clear()
	{
		m_loss_vector.clear();
		m_accuracy_vector.clear();
	}

	virtual neuro_u32 GetTotalScrollDataCount() const override;

	bool IsValid(neuro_64 nStart, neuro_64 nCount) override;

	bool GetViewData(neuro_64 nStart, neuro_64 nCount, neuro_u32 max_ylabel, neuro_64 cur_data_pos, gui::_graph_frame& graphFrame) override;
	CString GetDataTooltipLabel(neuro_u32 iGraph, neuro_u32 i) const override;

private:
	bool m_has_accuracy;

	std::vector<neuro_float> m_loss_vector;
	std::vector<neuro_float> m_accuracy_vector;
};

class CLossGraphWnd : public gui::win32::CGraphWnd
{
public:
	CLossGraphWnd();
	virtual ~CLossGraphWnd();

	void SetHasAccuracy(bool has) { m_source.SetHasAccuracy(has); }

	void AddData(neuro_float loss, neuro_float accuracy);
	void SetData(const std::vector<neuro_float>& loss_vector, const std::vector<neuro_float>& accuracy_vector);

	void Clear()
	{
		m_source.Clear();
	}
private:
	LossGraphSource m_source;
};
