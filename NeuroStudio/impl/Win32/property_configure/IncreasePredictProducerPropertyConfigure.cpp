#include "stdafx.h"
#include "DataProducerPropertyConfigure.h"

#include "NeuroData/model/IncreasePredictProducerModel.h"

using namespace property;

void IncreasePredictProducerPropertyConfigure::CompositeProperties(CMFCPropertyGridCtrl& list_ctrl, AbstractProducerModel* model)
{
	IncreasePredictProducerModel* producer = (IncreasePredictProducerModel*)model;

	CMFCPropertyGridProperty* column_prop = new CMFCPropertyGridProperty(L"Predict column index", (_variant_t)producer->GetSourceColumn()
		, L"Index of source column to predict", (DWORD_PTR)_prop_type::src_column);
	list_ctrl.AddProperty(column_prop);

	CMFCPropertyGridProperty* ma_prop = new CMFCPropertyGridProperty(L"MA", (_variant_t)producer->GetMovingAvarage()
		, L"Moving Average", (DWORD_PTR)_prop_type::src_ma);
	list_ctrl.AddProperty(ma_prop);

	if (producer->GetInput())
	{
		column_prop->EnableSpinControl(TRUE, 0, producer->GetInput()->GetColumnCount() - 1);
		ma_prop->EnableSpinControl(TRUE, 1, neuro_last16);
	}
	else
	{
		column_prop->Enable(FALSE);
		ma_prop->Enable(FALSE);
	}

	CMFCPropertyGridProperty* prop = new CMFCPropertyGridProperty(L"Distance", (_variant_t)producer->GetPredictDistance()
		, L"Predict distance", (DWORD_PTR)_prop_type::distance);
	list_ctrl.AddProperty(prop);

	const _increase_predict_type predict_type = producer->GetPredictType();
	prop = new CMFCPropertyGridProperty(L"Predict type", (_variant_t)ToString(predict_type)
		, L"Increase predict type. predict by value or rate", (DWORD_PTR)_prop_type::increase_predict_type);
	for (neuro_u32 i = 0; i < _countof(_increase_predict_type_string); i++)
		prop->AddOption(_increase_predict_type_string[i]);
	prop->AllowEdit(FALSE);
	list_ctrl.AddProperty(prop);

	const _predict_range_vector& range_vector = producer->GetRanges();
	CMFCPropertyGridProperty* range_count_prop = new CMFCPropertyGridProperty(L"Predict Range count", (_variant_t)range_vector.size()
		, L"", (DWORD_PTR)_prop_type::predict_range_count);
	list_ctrl.AddProperty(range_count_prop);

	CMFCPropertyGridProperty* range_list_prop = new CMFCPropertyGridProperty(L"Predict List");
	list_ctrl.AddProperty(range_list_prop);

	for (neuro_u32 i = 0; i < range_vector.size(); i++)
	{
		CMFCPropertyGridProperty* range_prop = new CMFCPropertyGridProperty(util::StringUtil::Format(L"%u Range", i).c_str());
		range_list_prop->AddSubItem(range_prop);

		CModelGridProperty* value_prop;
		if (predict_type == _increase_predict_type::rate)
		{
			value_prop = new CModelGridProperty(L"Increase rate[%]", (_variant_t)range_vector[i].value
				, L"Value is reference increasing percentage to compare", (DWORD_PTR)_prop_type::compare_value);
		}
		else
		{
			value_prop = new CModelGridProperty(L"Reference value", (_variant_t)range_vector[i].value
				, L"Value is reference value to compare", (DWORD_PTR)_prop_type::compare_value);
		}
		value_prop->index = i;
		range_prop->AddSubItem(value_prop);

		value_prop = new CModelGridProperty(L"Inequality", (_variant_t)ToString(range_vector[i].ineuality)
			, L"Compare type. if this is '>' and above(Increase percentage) is 30, target > ( 1 + 0.3 ) x source"
			, (DWORD_PTR)_prop_type::compare_type);
		value_prop->index = i;
		for (neuro_u32 i = 0; i < _countof(_inequality_type_string); i++)
			value_prop->AddOption(_inequality_type_string[i]);
		value_prop->AllowEdit(FALSE);

		range_prop->AddSubItem(value_prop);
	}
}
void IncreasePredictProducerPropertyConfigure::PropertyChanged(CModelGridProperty* prop, AbstractProducerModel* model, bool& reload) const
{
	IncreasePredictProducerModel* producer = (IncreasePredictProducerModel*)model;
	_prop_type prop_type = (_prop_type)prop->GetData();
	if(prop_type==_prop_type::src_column)
		producer->SetSourceColumn(prop->GetValue().intVal);
	else if (prop_type == _prop_type::src_ma)
		producer->SetMovingAvarage(prop->GetValue().intVal);
	else if (prop_type == _prop_type::distance)
		producer->SetPredictDistance(prop->GetValue().intVal);
	else if (prop_type == _prop_type::increase_predict_type)
	{
		CString value = prop->GetValue();
		for (neuro_u32 i = 0; i < _countof(_increase_predict_type_string); i++)
		{
			if (value == _increase_predict_type_string[i])
			{
				producer->SetPredictType((_increase_predict_type)i);
				reload = true;
				break;
			}
		}
	}
	else
	{
		_predict_range_vector ranges = producer->GetRanges();
		if(prop_type == _prop_type::predict_range_count)
			ranges.resize(prop->GetValue().intVal, { 1, _inequality_type::greater });
		else if (prop_type == _prop_type::compare_value)
			ranges[prop->index].value = prop->GetValue().fltVal;
		else if (prop_type == _prop_type::compare_type)
		{
			CString value = prop->GetValue();
			for (neuro_u32 i = 0; i < _countof(_inequality_type_string); i++)
			{
				if (value == _increase_predict_type_string[i])
				{
					ranges[prop->index].ineuality = (_inequality_type)i;
					break;
				}
			}
		}
		producer->SetRanges(ranges);
	}
}
