#include "stdafx.h"

#include "NetworkPropertyConfigure.h"

#include "desc/LayerDesc.h"
#include "desc/TensorShapeDesc.h"

using namespace property;

#include "NeuroKernel/network/NeuralNetwork.h"

void NetworkPropertyConfigure::CompositeProperties()
{
	if (m_network == NULL)
		return;

	const np::network::_LEARNING_INFO& info = m_network->GetLearningInfo();

	CMFCPropertyGridProperty* optimizer_prop = new CMFCPropertyGridProperty(L"Learning information");
	m_list_ctrl.AddProperty(optimizer_prop);

	CMFCPropertyGridProperty* prop = new CMFCPropertyGridProperty(L"Optimizer"
		, (variant_t)ToString(info.optimizer_type)
		, L"a Function to optimize weights"
		, (DWORD_PTR)&info.optimizer_type);
	for (neuro_u32 i = 0; i < _countof(optimizer_type_string); i++)
		prop->AddOption(optimizer_type_string[i]);
	prop->AllowEdit(FALSE);
	optimizer_prop->AddSubItem(prop);

	CMFCPropertyGridProperty* lr_prop = new CMFCPropertyGridProperty(L"Learn rate policy");
	optimizer_prop->AddSubItem(lr_prop);

	prop = new CMFCPropertyGridProperty(L"Learn rate"
		, (variant_t)info.optimizing_rule.lr_policy.lr_base
		, L"a Function to optimize weights"
		, (DWORD_PTR)&info.optimizing_rule.lr_policy.lr_base);
	lr_prop->AddSubItem(prop);

	prop = new CMFCPropertyGridProperty(L"Learning rate policy"
		, (variant_t)ToString(info.optimizing_rule.lr_policy.type)
		, L"a policy how learning rate is being changed"
		, (DWORD_PTR)&info.optimizing_rule.lr_policy.type);
	for (neuro_u32 i = 0; i < _countof(lr_policy_type_string); i++)
		prop->AddOption(lr_policy_type_string[i]);
	prop->AllowEdit(FALSE);
	lr_prop->AddSubItem(prop);

	prop = new CMFCPropertyGridProperty(L"gamma"
		, (variant_t)info.optimizing_rule.lr_policy.gamma
		, L""
		, (DWORD_PTR)&info.optimizing_rule.lr_policy.gamma);
	lr_prop->AddSubItem(prop);

	prop = new CMFCPropertyGridProperty(L"step"
		, (variant_t)info.optimizing_rule.lr_policy.step
		, L""
		, (DWORD_PTR)&info.optimizing_rule.lr_policy.step);
	lr_prop->AddSubItem(prop);

	CMFCPropertyGridProperty* wn_prop = new CMFCPropertyGridProperty(L"Weight normalization policy");
	optimizer_prop->AddSubItem(wn_prop);

	prop = new CMFCPropertyGridProperty(L"Type"
		, (variant_t)ToString(info.optimizing_rule.wn_policy.type)
		, L""
		, (DWORD_PTR)&info.optimizing_rule.wn_policy.type);
	for (neuro_u32 i = 0; i < _countof(wn_policy_type_string); i++)
		prop->AddOption(wn_policy_type_string[i]);
	prop->AllowEdit(FALSE);
	wn_prop->AddSubItem(prop);

	if (info.optimizing_rule.wn_policy.type != _wn_policy_type::none)
	{
		prop = new CMFCPropertyGridProperty(L"Weight decay"
			, (variant_t)info.optimizing_rule.wn_policy.weight_decay
			, L""
			, (DWORD_PTR)&info.optimizing_rule.wn_policy.weight_decay);
		wn_prop->AddSubItem(prop);
	}

	m_list_ctrl.ExpandAll();
}

void NetworkPropertyConfigure::PropertyChanged(CModelGridProperty* prop, bool& reload) const
{
	void* data = (void*)prop->GetData();

	np::network::_LEARNING_INFO& info = m_network->GetLearningInfo();
	if (data == &info.optimizer_type)
	{
		CString value = prop->GetValue();
		for (neuro_u32 i = 0; i < _countof(optimizer_type_string); i++)
		{
			if (value == optimizer_type_string[i])
			{
				info.optimizer_type = (_optimizer_type)i;

				// 원래 여기에서 모든 weight 초기화(?) 및 history, optimizer parameter들을 초기화해야한다.
				break;
			}
		}
	}
	else if (data == &info.optimizing_rule.lr_policy.lr_base)
	{
		info.optimizing_rule.lr_policy.lr_base = prop->GetValue().fltVal;
	}
	else if (data == &info.optimizing_rule.lr_policy.type)
	{
		CString value = prop->GetValue();
		for (neuro_u32 i = 0; i < _countof(lr_policy_type_string); i++)
		{
			if (value == lr_policy_type_string[i])
			{
				info.optimizing_rule.lr_policy.type = (_lr_policy_type)i;
				break;
			}
		}
	}
	else if (data == &info.optimizing_rule.lr_policy.gamma)
	{
		info.optimizing_rule.lr_policy.gamma = prop->GetValue().fltVal;
	}
	else if (data == &info.optimizing_rule.lr_policy.step)
	{
		info.optimizing_rule.lr_policy.step = prop->GetValue().uintVal;
	}
	else if (data == &info.optimizing_rule.wn_policy.type)
	{
		CString value = prop->GetValue();
		for (neuro_u32 i = 0; i < _countof(wn_policy_type_string); i++)
		{
			if (value == wn_policy_type_string[i])
			{
				info.optimizing_rule.wn_policy.type = (_wn_policy_type)i;
				break;
			}
		}
	}
	else if (data == &info.optimizing_rule.wn_policy.weight_decay)
	{
		info.optimizing_rule.wn_policy.weight_decay = prop->GetValue().fltVal;
	}
	else if (data == &info.data_batch_type)
	{
		CString value = prop->GetValue();
		for (neuro_u32 i = 0; i < _countof(train_data_batch_type_string); i++)
		{
			if (value == train_data_batch_type_string[i])
			{
				info.data_batch_type = (_train_data_batch_type)i;
				break;
			}
		}
	}
}
