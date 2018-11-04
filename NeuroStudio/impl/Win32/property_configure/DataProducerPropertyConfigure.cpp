#include "stdafx.h"

#include "DataProducerPropertyConfigure.h"

using namespace property;

NumericProducerPropertyConfigure numeric_config;
IncreasePredictProducerPropertyConfigure increase_predict_config;
NlpProducerPropertyConfigure nlp_config;
MnistProducerPropertyConfigure mnist_config;
ImageProducerPropertyConfigure image_config;

SubProducerPropertyConfigure* DataProducerPropertyConfigure::GetConfigure() const
{
	switch (GetModel()->GetProducerType())
	{
	case _producer_type::numeric:
		return &numeric_config;
	case _producer_type::increase_predict:
		return &increase_predict_config;
	case _producer_type::nlp:
		return &nlp_config;
	case _producer_type::mnist_img:
	case _producer_type::mnist_label:
		return &mnist_config;
	case _producer_type::image_file:
		return &image_config;
		break;
	case _producer_type::imagenet:
		break;
	case _producer_type::cifar:
		break;
	}
	return NULL;
}

void DataProducerPropertyConfigure::CompositeProperties()
{
	m_start_prop = m_labelout_count_prop = m_label_dir_prop = NULL;

	SubProducerPropertyConfigure* configure = GetConfigure();
	if (!configure)
		return;

	AbstractProducerModel* model = GetModel();
	configure->CompositeProperties(m_list_ctrl, model);

	m_start_prop = new CModelGridProperty(L"Start", (_variant_t)model->GetStartPosition(), L"Start position to read data");
	m_start_prop->EnableSpinControl(TRUE, model->GetAvailableStartPosition());
	m_list_ctrl.AddProperty(m_start_prop);

	CModelGridProperty* tensor_property = new CModelGridProperty(L"Tensor", (_variant_t)str_rc::TensorShapeDesc::GetDataShapeText(model->GetDataShape()).c_str()
		, L"final data tensor");
	tensor_property->AllowEdit(FALSE);
	m_list_ctrl.AddProperty(tensor_property);

	if(model->GetLabelOutType() == _label_out_type::label_dir)
	{
		m_labelout_count_prop = new CModelGridProperty(L"Label count"
			, (_variant_t)model->GetLabelOutCount()
			, L"Count of output label value.\r\nIf it is less than 2, this has no label");
		m_labelout_count_prop->EnableSpinControl(TRUE);
		m_list_ctrl.AddProperty(m_labelout_count_prop);

		const std_string_vector& label_dir_vector = model->GetLabelDirVector();
		if (label_dir_vector.size() > 0)
		{
			m_label_dir_prop = new CModelGridProperty(L"Label directory");
			m_list_ctrl.AddProperty(m_label_dir_prop);

			for (neuro_u32 i = 0, n = label_dir_vector.size(); i < n; i++)
			{
				CModelGridProperty* scope_prop = new CModelGridProperty(util::StringUtil::Format(L"%u label", i).c_str()
					, (variant_t)util::StringUtil::MultiByteToWide(label_dir_vector[i]).c_str(), L"");
				scope_prop->index = i;
				m_label_dir_prop->AddSubItem(scope_prop);
			}
		}
	}
	m_list_ctrl.ExpandAll();
}

void DataProducerPropertyConfigure::PropertyChanged(CModelGridProperty* prop, bool& reload) const
{
	if (prop == m_start_prop)
	{
		GetModel()->SetStartPosition(prop->GetValue().uintVal);
		return;
	}
	else if (prop == m_labelout_count_prop)
	{
		if (prop->GetValue().uintVal < 2)
			prop->SetValue((_variant_t)0);

		GetModel()->SetLabelOutCount(prop->GetValue().uintVal);
		reload = true;
		return;
	}
	else if (prop->GetParent() == m_label_dir_prop)
	{
		AbstractProducerModel* model = GetModel();

		std_string_vector label_dir_vector = model->GetLabelDirVector();

		CString value = prop->GetValue();
		label_dir_vector[prop->index] = util::StringUtil::WideToMultiByte((const wchar_t*)value);

		model->SetLabelDirVector(label_dir_vector);
		return;
	}

	SubProducerPropertyConfigure* configure = GetConfigure();
	if (!configure)
		return;

	configure->PropertyChanged(prop, GetModel(), reload);
}
