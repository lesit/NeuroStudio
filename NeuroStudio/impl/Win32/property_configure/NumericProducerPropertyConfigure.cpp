#include "stdafx.h"

#include "DataProducerPropertyConfigure.h"

#include "NeuroData/model/NumericProducerModel.h"

using namespace property;

void NumericProducerPropertyConfigure::CompositeProperties(CMFCPropertyGridCtrl& list_ctrl, AbstractProducerModel* model)
{
	NumericProducerModel* producer = (NumericProducerModel*)model;
	const std::map<neuro_u32, neuro_u32>& using_index_ma_map = producer->GetUsingSourceColumns();

	CMFCPropertyGridProperty* onehot_prop = new CMFCPropertyGridProperty(L"Onehot size", (_variant_t)producer->GetLabelOutCount()
		, L"Onehot encoding size. If you set this value over 1, data will be index of max value in using columns"
		, (DWORD_PTR)_prop_type::onehot);
	list_ctrl.AddProperty(onehot_prop);

	if (producer->GetLabelOutCount() > 1)
	{
		neuro_u32 source_column = 0;
		neuro_u32 ma = 1;
		if (using_index_ma_map.size() > 0)
		{
			source_column = using_index_ma_map.begin()->first;
			ma = using_index_ma_map.begin()->second;
		}
		CMFCPropertyGridProperty* index_prop = new CMFCPropertyGridProperty(L"Column index", (_variant_t)source_column
			, L"Using column index in source columns", (DWORD_PTR)_prop_type::column_index);
		list_ctrl.AddProperty(index_prop);

		CMFCPropertyGridProperty* ma_prop = new CMFCPropertyGridProperty(L"MA", (_variant_t)ma
			, L"Moving Average", (DWORD_PTR)_prop_type::src_ma);
		list_ctrl.AddProperty(ma_prop);

		if (producer->GetInput())
		{
			index_prop->EnableSpinControl(TRUE, 0, producer->GetInput()->GetColumnCount() - 1);
			ma_prop->EnableSpinControl(TRUE, 1, neuro_last16);
		}
		else 
		{
			index_prop->Enable(FALSE);
			ma_prop->Enable(FALSE);
		}
	}
	else
	{
		CMFCPropertyGridProperty* column_list_prop = new CMFCPropertyGridProperty(L"Column list");
		list_ctrl.AddProperty(column_list_prop);
		if (producer->GetInput())
		{
			neuro_u32 input_column_count = producer->GetInput()->GetColumnCount();
			for (neuro_u32 i = 0; i < input_column_count; i++)
			{
				CModelGridProperty* column_prop = new CModelGridProperty(util::StringUtil::Format<wchar_t>(L"%u column", i).c_str());
				column_list_prop->AddSubItem(column_prop);

				neuro_u32 ma = 1;
				bool use = false;
				std::map<neuro_u32, neuro_u32>::const_iterator using_it = using_index_ma_map.find(i);
				if (using_it != using_index_ma_map.end())
				{
					ma = using_it->second;
					use = true;
				}

				CModelGridProperty* prop = new CModelGridProperty(L"Use", (_variant_t)use
					, L"Select using or not", (DWORD_PTR)_prop_type::src_use);
				prop->index = i;
				column_prop->AddSubItem(prop);

				prop = new CModelGridProperty(L"MA", (_variant_t)ma
					, L"Moving Average", (DWORD_PTR)_prop_type::src_ma);
				prop->index = i;
				if (use)
					prop->EnableSpinControl(TRUE, 1, neuro_last16);
				else
					prop->Enable(FALSE);

				column_prop->AddSubItem(prop);
			}
		}
	}
}

void NumericProducerPropertyConfigure::PropertyChanged(CModelGridProperty* prop, AbstractProducerModel* model, bool& reload) const
{
	NumericProducerModel* producer = (NumericProducerModel*)model;

	const std::map<neuro_u32, neuro_u32>& using_index_ma_map = producer->GetUsingSourceColumns();
	if (producer->GetLabelOutCount() > 0)
	{
		if ((_prop_type)prop->GetData() == _prop_type::onehot)
		{
			producer->SetLabelOutCount(prop->GetValue().intVal);
			reload = true;
			return;
		}

		neuro_u32 source_column = 0;
		neuro_u32 ma = 1;
		if (using_index_ma_map.size() > 0)
		{
			source_column = using_index_ma_map.begin()->first;
			ma = using_index_ma_map.begin()->second;
		}

		if ((_prop_type)prop->GetData() == _prop_type::column_index)
			source_column = prop->GetValue().intVal;
		else if ((_prop_type)prop->GetData() == _prop_type::src_ma)
			ma = prop->GetValue().intVal;
		else
			return;
		producer->InsertSourceColumn(source_column, ma);
	}
	else
	{
		if ((_prop_type)prop->GetData() == _prop_type::src_use)
		{
			if (prop->GetValue().boolVal)
			{ 
				neuro_u32 ma = 1;
				if (using_index_ma_map.find(prop->index) != using_index_ma_map.end())
					ma = using_index_ma_map.find(prop->index)->second;
				producer->InsertSourceColumn(prop->index, ma);
			}
			else
				producer->EraseSourceColumn(prop->index);

			reload = true;
		}
		else if ((_prop_type)prop->GetData() == _prop_type::src_ma)
		{
			producer->InsertSourceColumn(prop->index, prop->GetValue().intVal);
		}
	}
}
