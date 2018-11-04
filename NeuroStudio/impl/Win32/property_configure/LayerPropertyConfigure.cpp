#include "stdafx.h"

#include "LayerPropertyConfigure.h"

#include "desc/LayerDesc.h"
#include "desc/TensorShapeDesc.h"

using namespace property;
using namespace np::network;

std::vector<neuro_u32> LayerPropertyConfigure::GetSubTypeVector() const
{
	if (m_layer->GetLayerType() == _layer_type::input || m_layer->GetLayerType() == _layer_type::output)
		return{ (neuro_u32)m_layer->GetLayerType() };

	return{ (neuro_u32)_layer_type::fully_connected
			, (neuro_u32)_layer_type::convolutional
			, (neuro_u32)_layer_type::pooling
			, (neuro_u32)_layer_type::dropout
			, (neuro_u32)_layer_type::concat
			, (neuro_u32)_layer_type::rnn
			, (neuro_u32)_layer_type::batch_norm };
}

const wchar_t* LayerPropertyConfigure::GetSubTypeString(neuro_u32 type) const
{
	return ToString((_layer_type)type);
}

#include "../DesignNetworkWnd.h"

bool LayerPropertyConfigure::SubTypeChange(AbstractBindedViewManager* view, neuro_u32 type)
{
	if ((_layer_type)type == m_layer->GetLayerType())
		return false;

	if (m_layer->GetLayerType() == _layer_type::input || m_layer->GetLayerType() == _layer_type::output)
		return false;

	m_last_set_entries.SetEntry(m_layer->GetLayerType(), ((HiddenLayer*)m_layer)->GetEntry());	// 마지막 entry 를 저장해 두자.
	const nsas::_LAYER_STRUCTURE_UNION& entry = m_last_set_entries.GetEntry((_layer_type)type);

	// 만약 바꾸기 전의 layer type이 원래의 layer type(LayerPropertyConfigure 최초 생성 또는 ChangeLayer로 의해 바뀌었을때) 일때
	// 삭제되는 입력들에 대한 정보를 가지고 있어야, 다시 원래의 layer type으로 바꾸었을때 넣어줄수 있다.
	// 즉, 다른 layer 또는 preprocessor나 없는걸로 바꾸기 전에는 기존 입력을 보존해서 다시 연결하는 불편을 해소해준다.
	_slice_input_vector* org_erased_input_vector = m_layer->GetLayerType() == m_org_type ? &m_org_erased_input_vector : NULL;

	if(!((DesignNetworkWnd*)view)->ChangeLayerType((HiddenLayer*)m_layer, (_layer_type)type, &entry, org_erased_input_vector))
		return false;

	if (m_layer->GetLayerType() == m_org_type)
	{
		// 다시 원래의 layer type으로 바꾸었을때 삭제된 입력들 복구시킨다.
		if (((HiddenLayer*)m_layer)->BatchAppendInputs(m_org_erased_input_vector))
		{
			((DesignNetworkWnd*)view)->LoadView();
		}
		else
			DEBUG_OUTPUT(L"failed add original input");
	}
	return true;
}

InputLayerPropertyConfigure input_config;
FcLayerPropertyConfigure fc_config;
ConvLayerPropertyConfigure conv_config;
PoolLayerPropertyConfigure pool_config;
DropoutLayerPropertyConfigure dropout_config;
BnLayerPropertyConfigure bn_config;
ConcatLayerPropertyConfigure concat_config;
RnnLayerPropertyConfigure rnn_config;
OutputLayerPropertyConfigure output_config;

SubLayerPropertyConfigure* LayerPropertyConfigure::GetConfigure() const
{
	switch (m_layer->GetLayerType())
	{
	case _layer_type::input:
		return &input_config;
	case _layer_type::fully_connected:
		return &fc_config;
	case _layer_type::convolutional:
		return &conv_config;
	case _layer_type::pooling:
		return &pool_config;
	case _layer_type::dropout:
		return &dropout_config;
	case _layer_type::batch_norm:
		return &bn_config;
	case _layer_type::concat:
		return &concat_config;
	case _layer_type::rnn:
		return &rnn_config;
	case _layer_type::output:
		return &output_config;
	}
	return NULL;
}

void LayerPropertyConfigure::CompositeProperties()
{
	SubLayerPropertyConfigure* config = GetConfigure();
	if (config == NULL)
		return;

	// 모든 입력 list에 추가
	if (m_layer->GetLayerType() != _layer_type::input)
	{
		CMFCPropertyGridProperty* io_prop = new CMFCPropertyGridProperty(L"Input & Output");
		m_list_ctrl.AddProperty(io_prop);

		const _slice_input_vector& input_vector = ((HiddenLayer*)m_layer)->GetInputVector();

		std::wstring input_list_str = util::StringUtil::Format(L"Input list : %u", input_vector.size());
		CMFCPropertyGridProperty* input_list_prop = new CMFCPropertyGridProperty(input_list_str.c_str());
		io_prop->AddSubItem(input_list_prop);

		for (neuro_u32 i = 0; i < input_vector.size(); i++)
		{
			const _SLICE_INPUT& input = input_vector[i];
			MATRIX_POINT mp = m_view.GetLayerLocation(*input.layer);

			std::wstring layer_desc = str_rc::LayerDesc::GetSimpleName(*input.layer);
			layer_desc += util::StringUtil::Format(L" [level:%u, row:%u]", mp.level, mp.row);
			CMFCPropertyGridProperty* input_prop = new CMFCPropertyGridProperty(layer_desc.c_str());
			input_list_prop->AddSubItem(input_prop);

			CMFCPropertyGridProperty* prop = new CMFCPropertyGridProperty(L"Tensor"
				, (variant_t)str_rc::TensorShapeDesc::GetTensorText(input.GetTensor()).c_str(), L"");
			prop->Enable(FALSE);
			input_prop->AddSubItem(prop);
			// slice 기능을 만들면 여기에도 추가하자
		}

		CMFCPropertyGridProperty* out_ts_prop = new CMFCPropertyGridProperty(L"Output tensor"
			, (variant_t)str_rc::TensorShapeDesc::GetTensorText(m_layer->GetOutTensorShape()).c_str(), L"");
		out_ts_prop->Enable(FALSE);
		io_prop->AddSubItem(out_ts_prop);

		if (((HiddenLayer*)m_layer)->HasActivation())
		{
			CMFCPropertyGridProperty* activation_prop = new CMFCPropertyGridProperty(L"Activation"
				, (variant_t)ToString(((HiddenLayer*)m_layer)->GetActivation()), L"");
			if (((HiddenLayer*)m_layer)->AvailableChangeActivation())
			{
				for (neuro_u32 i = 0; i < _countof(activation_type_string); i++)
					activation_prop->AddOption(activation_type_string[i]);
			}
			activation_prop->AllowEdit(FALSE);

			m_list_ctrl.AddProperty(activation_prop);
		}
		if (((HiddenLayer*)m_layer)->HasWeight())
		{
			auto _add_weight_property = [&](const wchar_t* name, const wchar_t* multlr_desc, const _LAYER_WEIGHT_INFO* info)
			{
				CMFCPropertyGridProperty* group_prop = new CMFCPropertyGridProperty(name, (DWORD_PTR)info);
				m_list_ctrl.AddProperty(group_prop);

				CMFCPropertyGridProperty* prop = new CMFCPropertyGridProperty(L"Init type"
					, (variant_t)ToString(info->init_type), L"", (DWORD_PTR)&info->init_type);
				for (neuro_u32 i = 0; i < _countof(weight_init_type_string); i++)
					prop->AddOption(weight_init_type_string[i]);
				prop->AllowEdit(FALSE);

				group_prop->AddSubItem(prop);

				/*	이건 나중에
				prop = new CMFCPropertyGridProperty(L"Init factor"
					, (variant_t)info->init_scale, L"", (DWORD_PTR)&info->init_scale);
				group_prop->AddSubItem(prop);
				*/
				prop = new CMFCPropertyGridProperty(L"Mult lr"
					, (variant_t)info->mult_lr, multlr_desc, (DWORD_PTR)&info->mult_lr);
				group_prop->AddSubItem(prop);

				prop = new CMFCPropertyGridProperty(L"Decay"
					, (variant_t)info->decay, L"", (DWORD_PTR)&info->decay);
				group_prop->AddSubItem(prop);
			};

			const _LAYER_WEIGHT_INFO* weight_info = ((HiddenLayer*)m_layer)->GetWeightInfo(_layer_data_type::weight);
			const _LAYER_WEIGHT_INFO* bias_info = ((HiddenLayer*)m_layer)->GetWeightInfo(_layer_data_type::bias);
			if (weight_info)
				_add_weight_property(L"Weight info", L"Usually 1.0", weight_info);
			if (bias_info)
				_add_weight_property(L"Bias info", L"Usually 2.0", bias_info);
		}
	}
	// layer 속성 추가
	config->CompositeProperties(m_list_ctrl, m_layer);

	m_list_ctrl.ExpandAll();
}

void LayerPropertyConfigure::PropertyChanged(CModelGridProperty* prop, bool& reload) const
{
	if (wcscmp(prop->GetName(), L"Activation") == 0)
	{
		CString str = prop->GetValue();
		for (neuro_u32 i = 0; i < _countof(activation_type_string); i++)
		{
			if (str == activation_type_string[i])
				((HiddenLayer*)m_layer)->SetActivation((_activation_type)i);
		}
		return;
	}

	{
		const _LAYER_WEIGHT_INFO* weight_info = ((HiddenLayer*)m_layer)->GetWeightInfo(_layer_data_type::weight);
		const _LAYER_WEIGHT_INFO* bias_info = ((HiddenLayer*)m_layer)->GetWeightInfo(_layer_data_type::bias);
		if (prop->GetParent() && ((void*)prop->GetParent()->GetData() == weight_info || (void*)prop->GetParent()->GetData() == bias_info))
		{
			void* data = (void*)prop->GetData();

			if (data == &weight_info->init_type || data == &bias_info->init_type)
			{
				CString value = prop->GetValue();
				for (neuro_u32 i = 0; i < _countof(weight_init_type_string); i++)
				{
					if (value == weight_init_type_string[i])
					{
						*((network::_weight_init_type*)data) = (network::_weight_init_type)i;
						break;
					}
				}
			}
			else
				*((neuro_float*)data) = prop->GetValue().fltVal;

			return;
		}
	}
	{
		SubLayerPropertyConfigure* config = GetConfigure();
		if (config == NULL)
			return;

		config->PropertyChanged(prop, m_layer, reload);

		if(m_layer->GetLayerType()!=_layer_type::input)
			((HiddenLayer*)m_layer)->CheckChangedEntry();
	}
}

void InputLayerPropertyConfigure::CompositeProperties(CMFCPropertyGridCtrl& list_ctrl, AbstractLayer* layer)
{
	const tensor::TensorShape& ts = layer->GetOutTensorShape();

	CMFCPropertyGridProperty* prop = new CMFCPropertyGridProperty(L"Time length", (variant_t)ts.time_length
		, L"This indicate a length of grouping by time. example, video can be read by several image frame, so time length is the frame count"
		, (DWORD_PTR) &ts.time_length);
	prop->EnableSpinControl(TRUE, 1, neuro_last31);
	list_ctrl.AddProperty(prop);

	prop = new CMFCPropertyGridProperty(L"Channel", (variant_t)ts.GetChannelCount()
		, L"This nomaly indicate image color depth. Mono image is 1, Color image is 3(red, green, blue). Also, You can define first dimension of 3D");
	prop->Enable(FALSE);
	list_ctrl.AddProperty(prop);

	prop = new CMFCPropertyGridProperty(L"Height", (variant_t)ts.GetHeight()
		, L"Height when 2D or 3D image. Also, First dimension of 2D");
	prop->Enable(FALSE);
	list_ctrl.AddProperty(prop);

	prop = new CMFCPropertyGridProperty(L"Width", (variant_t)ts.GetWidth()
		, L"Width when 2D or 3D image. Also, Second dimension of 2D");
	prop->Enable(FALSE);
	list_ctrl.AddProperty(prop);
}

void InputLayerPropertyConfigure::PropertyChanged(CModelGridProperty* prop, AbstractLayer* layer, bool& reload)
{
	*((neuro_u32*)prop->GetData()) = prop->GetValue().intVal;
}

void FcLayerPropertyConfigure::CompositeProperties(CMFCPropertyGridCtrl& list_ctrl, AbstractLayer* layer)
{
	const np::nsas::_LAYER_STRUCTURE_UNION& entry = ((HiddenLayer*)layer)->GetEntry();

	CMFCPropertyGridProperty* prop = new CMFCPropertyGridProperty(L"Output count", (variant_t)entry.fc.output_count
		, L"", (DWORD_PTR)&entry.fc.output_count);
	list_ctrl.AddProperty(prop);
}

void FcLayerPropertyConfigure::PropertyChanged(CModelGridProperty* prop, AbstractLayer* layer, bool& reload)
{
	*((neuro_u32*)prop->GetData()) = prop->GetValue().intVal;

	reload = true;
}

#include "NeuroKernel/network/ConvLayerConfigure.h"
void ConvLayerPropertyConfigure::CompositeProperties(CMFCPropertyGridCtrl& list_ctrl, AbstractLayer* layer)
{
	const np::nsas::_LAYER_STRUCTURE_UNION& entry = ((HiddenLayer*)layer)->GetEntry();

	CMFCPropertyGridProperty* kernel_prop = new CMFCPropertyGridProperty(L"Filter");
	list_ctrl.AddProperty(kernel_prop);

	CMFCPropertyGridProperty* prop = new CMFCPropertyGridProperty(L"Channel", (_variant_t) entry.conv.channel_count, L""
		, (DWORD_PTR)&entry.conv.channel_count);
	kernel_prop->AddSubItem(prop);
	prop = new CMFCPropertyGridProperty(L"Height", (_variant_t)entry.conv.filter.kernel_height, L""
		, (DWORD_PTR)&entry.conv.filter.kernel_height);
	kernel_prop->AddSubItem(prop);
	prop = new CMFCPropertyGridProperty(L"Width", (_variant_t)entry.conv.filter.kernel_width, L""
		, (DWORD_PTR)&entry.conv.filter.kernel_width);
	kernel_prop->AddSubItem(prop);

	CMFCPropertyGridProperty* dilation_prop = new CMFCPropertyGridProperty(L"Dilation");
	list_ctrl.AddProperty(dilation_prop);

	prop = new CMFCPropertyGridProperty(L"Height", (_variant_t)entry.conv.dilation_height, L""
		, (DWORD_PTR)&entry.conv.dilation_height);
	dilation_prop->AddSubItem(prop);
	prop = new CMFCPropertyGridProperty(L"Width", (_variant_t)entry.conv.dilation_width, L""
		, (DWORD_PTR)&entry.conv.dilation_width);
	dilation_prop->AddSubItem(prop);

	CMFCPropertyGridProperty* stride_prop = new CMFCPropertyGridProperty(L"Stride");
	list_ctrl.AddProperty(stride_prop);
	prop = new CMFCPropertyGridProperty(L"Height", (_variant_t)entry.conv.filter.stride_height, L""
		, (DWORD_PTR)&entry.conv.filter.stride_height);
	stride_prop->AddSubItem(prop);
	prop = new CMFCPropertyGridProperty(L"Width", (_variant_t)entry.conv.filter.stride_width, L""
		, (DWORD_PTR)&entry.conv.filter.stride_width);
	stride_prop->AddSubItem(prop);

	CMFCPropertyGridProperty* pad_prop = new CMFCPropertyGridProperty(L"Padding");
	list_ctrl.AddProperty(pad_prop);
	
	prop = new CMFCPropertyGridProperty(L"Type", (_variant_t) ToString((_pad_type) entry.conv.pad_type), L""
		, (DWORD_PTR)&entry.conv.pad_type);	// 변경되면 전체 출력 다시 해야함
	for (neuro_u32 i = 0; i < _countof(pad_type_string); i++)
		prop->AddOption(pad_type_string[i]);
	prop->AllowEdit(FALSE);
	pad_prop->AddSubItem(prop);

	BOOL allow_edit = (_pad_type)entry.conv.pad_type == _pad_type::user_define;

	std::pair<neuro_u32, neuro_u32> pad_h = ConvLayerConfigure::GetPad((HiddenLayer&)*layer, true);
	std::pair<neuro_u32, neuro_u32> pad_w = ConvLayerConfigure::GetPad((HiddenLayer&)*layer, false);

	prop = new CMFCPropertyGridProperty(L"Top pad", (_variant_t)util::StringUtil::Transform<wchar_t>(pad_h.first).c_str()
		, L"", (DWORD_PTR)&entry.conv.filter.pad_t);
	prop->AllowEdit(allow_edit);
	pad_prop->AddSubItem(prop);

	prop = new CMFCPropertyGridProperty(L"Bottom pad", (_variant_t)util::StringUtil::Transform<wchar_t>(pad_h.second).c_str(), L""
		, (DWORD_PTR)&entry.conv.filter.pad_b);
	prop->AllowEdit(allow_edit);
	pad_prop->AddSubItem(prop);

	prop = new CMFCPropertyGridProperty(L"Left pad", (_variant_t)util::StringUtil::Transform<wchar_t>(pad_w.first).c_str(), L""
		, (DWORD_PTR)&entry.conv.filter.pad_l);
	prop->AllowEdit(allow_edit);
	pad_prop->AddSubItem(prop);

	prop = new CMFCPropertyGridProperty(L"Right pad", (_variant_t)util::StringUtil::Transform<wchar_t>(pad_w.second).c_str(), L""
		, (DWORD_PTR)&entry.conv.filter.pad_r);
	prop->AllowEdit(allow_edit);
	pad_prop->AddSubItem(prop);
}

void ConvLayerPropertyConfigure::PropertyChanged(CModelGridProperty* prop, AbstractLayer* layer, bool& reload)
{
	const np::nsas::_LAYER_STRUCTURE_UNION& entry = ((HiddenLayer*)layer)->GetEntry();

	if ((DWORD_PTR)&entry.conv.pad_type == prop->GetData())
	{
		CString value = prop->GetValue();
		for (neuro_u32 i = 0; i < _countof(pad_type_string); i++)
		{
			if (value == pad_type_string[i])
			{
				*((neuro_u16*)prop->GetData()) = i;
				break;
			}
		}
	}
	else if((DWORD_PTR)&entry.conv.channel_count == prop->GetData())
		*((neuro_u32*)prop->GetData()) = prop->GetValue().intVal;
	else
		*((neuro_u16*)prop->GetData()) = prop->GetValue().intVal;

	reload = true;
}

void PoolLayerPropertyConfigure::CompositeProperties(CMFCPropertyGridCtrl& list_ctrl, AbstractLayer* layer)
{
	const np::nsas::_LAYER_STRUCTURE_UNION& entry = ((HiddenLayer*)layer)->GetEntry();

	CMFCPropertyGridProperty* prop = new CMFCPropertyGridProperty(L"Type", (_variant_t) ToString((_pooling_type)entry.pooling.type), L""
		, (DWORD_PTR)&entry.pooling.type);
	for (neuro_u32 i = 0; i < _countof(pooling_type_string); i++)
		prop->AddOption(pooling_type_string[i]);
	prop->AllowEdit(FALSE);
	list_ctrl.AddProperty(prop);

	CMFCPropertyGridProperty* kernel_prop = new CMFCPropertyGridProperty(L"Kernel");
	list_ctrl.AddProperty(kernel_prop);

	prop = new CMFCPropertyGridProperty(L"Height", (_variant_t)entry.pooling.filter.kernel_height, L""
		, (DWORD_PTR)&entry.pooling.filter.kernel_height);
	kernel_prop->AddSubItem(prop);
	prop = new CMFCPropertyGridProperty(L"Width", (_variant_t)entry.pooling.filter.kernel_width, L""
		, (DWORD_PTR)&entry.pooling.filter.kernel_width);
	kernel_prop->AddSubItem(prop);

	CMFCPropertyGridProperty* stride_prop = new CMFCPropertyGridProperty(L"Stride");
	list_ctrl.AddProperty(stride_prop);
	prop = new CMFCPropertyGridProperty(L"Height", (_variant_t)entry.pooling.filter.stride_height, L""
		, (DWORD_PTR)&entry.pooling.filter.stride_height);
	stride_prop->AddSubItem(prop);
	prop = new CMFCPropertyGridProperty(L"Width", (_variant_t)entry.pooling.filter.stride_width, L""
		, (DWORD_PTR)&entry.pooling.filter.stride_width);
	stride_prop->AddSubItem(prop);
}

void PoolLayerPropertyConfigure::PropertyChanged(CModelGridProperty* prop, AbstractLayer* layer, bool& reload)
{
	const np::nsas::_LAYER_STRUCTURE_UNION& entry = ((HiddenLayer*)layer)->GetEntry();

	if ((DWORD_PTR)&entry.pooling.type == prop->GetData())
	{
		CString value = prop->GetValue();
		for (neuro_u32 i = 0; i < _countof(pooling_type_string); i++)
		{
			if (value == pooling_type_string[i])
			{
				*((neuro_u16*)prop->GetData()) = i;
				break;
			}
		}
	}
	else
		*((neuro_u16*)prop->GetData()) = prop->GetValue().intVal;

	reload = true;
}

void DropoutLayerPropertyConfigure::CompositeProperties(CMFCPropertyGridCtrl& list_ctrl, AbstractLayer* layer)
{
	const np::nsas::_LAYER_STRUCTURE_UNION& entry = ((HiddenLayer*)layer)->GetEntry();

	CMFCPropertyGridProperty* prop = new CMFCPropertyGridProperty(L"Dropout rate(%)", (variant_t)neuro_u32(entry.dropout.dropout_rate * 100.f)
		, L"Dropout rate by percentage", (DWORD_PTR)&entry.dropout.dropout_rate);
	prop->EnableSpinControl(TRUE, 1, 99);
	list_ctrl.AddProperty(prop);
}

void DropoutLayerPropertyConfigure::PropertyChanged(CModelGridProperty* prop, AbstractLayer* layer, bool& reload)
{
	*((neuro_float*)prop->GetData()) = neuro_float(prop->GetValue().uintVal) / 100.f;
}

void BnLayerPropertyConfigure::CompositeProperties(CMFCPropertyGridCtrl& list_ctrl, AbstractLayer* layer)
{
	const np::nsas::_LAYER_STRUCTURE_UNION& entry = ((HiddenLayer*)layer)->GetEntry();

	CMFCPropertyGridProperty* prop = new CMFCPropertyGridProperty(L"Momentum", (variant_t)entry.batch_norm.momentum
		, L"", (DWORD_PTR)&entry.batch_norm.momentum);
	list_ctrl.AddProperty(prop);

	prop = new CMFCPropertyGridProperty(L"Epsilon", (variant_t)entry.batch_norm.eps
		, L"", (DWORD_PTR)&entry.batch_norm.eps);
	list_ctrl.AddProperty(prop);
}

void BnLayerPropertyConfigure::PropertyChanged(CModelGridProperty* prop, AbstractLayer* layer, bool& reload)
{
	*((neuro_float*)prop->GetData()) = prop->GetValue().fltVal;
}

#include "NeuroKernel/network/ConcatLayerConfigure.h"
void ConcatLayerPropertyConfigure::CompositeProperties(CMFCPropertyGridCtrl& list_ctrl, AbstractLayer* layer)
{
	const np::nsas::_LAYER_STRUCTURE_UNION& entry = ((HiddenLayer*)layer)->GetEntry();

	_CONCAT_INFO info = ConcatLayerConfigure::GetConcatInfo(((HiddenLayer&)*layer));

	CMFCPropertyGridProperty* concat_prop = new CMFCPropertyGridProperty(L"Concatenated");
	list_ctrl.AddProperty(concat_prop);

	CMFCPropertyGridProperty* prop;
	prop = new CMFCPropertyGridProperty(L"Axis point", (variant_t)(info.concat_axis < 0 ? L"time" : util::StringUtil::Transform<wchar_t>(info.concat_axis).c_str())
		, L"concatenated axis in input tensors");
	prop->Enable(FALSE);
	concat_prop->AddSubItem(prop);

	prop = new CMFCPropertyGridProperty(L"Join tensor", (variant_t)str_rc::TensorShapeDesc::GetTensorText(info.join_ts).c_str()
		, L"");
	prop->Enable(FALSE);
	concat_prop->AddSubItem(prop);

	prop = new CMFCPropertyGridProperty(L"Axis size", (variant_t)util::StringUtil::Transform<wchar_t>(info.concat_axis_size).c_str()
		, L"");
	prop->Enable(FALSE);
	concat_prop->AddSubItem(prop);

	prop = new CMFCPropertyGridProperty(L"Concat tensor", (variant_t)str_rc::TensorShapeDesc::GetTensorText(info.concat_ts).c_str()
		, L"");
	prop->Enable(FALSE);
	concat_prop->AddSubItem(prop);
}

void RnnLayerPropertyConfigure::CompositeProperties(CMFCPropertyGridCtrl& list_ctrl, AbstractLayer* layer)
{
	const np::nsas::_LAYER_STRUCTURE_UNION& entry = ((HiddenLayer*)layer)->GetEntry();

	CMFCPropertyGridProperty* prop = new CMFCPropertyGridProperty(L"Type", (_variant_t)ToString((_rnn_type)entry.rnn.type), L""
		, (DWORD_PTR)&entry.rnn.type);
	for (neuro_u32 i = 0; i < _countof(rnn_type_string); i++)
		prop->AddOption(rnn_type_string[i]);
	prop->AllowEdit(FALSE);
	list_ctrl.AddProperty(prop);

	prop = new CMFCPropertyGridProperty(L"None time input", (variant_t)(entry.rnn.is_non_time_input!=0)
		, L"a non time varying data as input. so, transfer input data into each LSTM units."
		, (DWORD_PTR)&entry.rnn.is_non_time_input);
	list_ctrl.AddProperty(prop);

	prop = new CMFCPropertyGridProperty(L"Output time", (variant_t)entry.rnn.fix_time_length
		, L"", (DWORD_PTR)&entry.rnn.fix_time_length);
	list_ctrl.AddProperty(prop);

	prop = new CMFCPropertyGridProperty(L"Output count", (variant_t)entry.rnn.output_count
		, L"", (DWORD_PTR)&entry.rnn.output_count);
	list_ctrl.AddProperty(prop);
}

void RnnLayerPropertyConfigure::PropertyChanged(CModelGridProperty* prop, AbstractLayer* layer, bool& reload)
{
	np::nsas::_LAYER_STRUCTURE_UNION entry = ((HiddenLayer*)layer)->GetEntry();

	void* data = (void*)prop->GetData();
	if (data == &entry.rnn.type)
	{
		CString value = prop->GetValue();
		for (neuro_u32 i = 0; i < _countof(rnn_type_string); i++)
		{
			if (value == rnn_type_string[i])
			{
				*((neuro_u16*)prop->GetData()) = i;
				break;
			}
		}
	}
	else if (data == &entry.rnn.is_non_time_input)
		*((neuro_u8*)data) = prop->GetValue().boolVal;
	else
		*((neuro_u32*)data) = prop->GetValue().uintVal;

	reload = true;
}

void OutputLayerPropertyConfigure::CompositeProperties(CMFCPropertyGridCtrl& list_ctrl, AbstractLayer* layer)
{
	const np::nsas::_LAYER_STRUCTURE_UNION& entry = ((HiddenLayer*)layer)->GetEntry();

	CMFCPropertyGridProperty* prop = new CMFCPropertyGridProperty(L"Loss type", (_variant_t)ToString((_loss_type)entry.output.loss_type)
		, L""
		, (DWORD_PTR)&entry.output.loss_type);
	for (neuro_u32 i = 0; i < _countof(loss_type_string); i++)
		prop->AddOption(loss_type_string[i]);
	prop->AllowEdit(FALSE);
	list_ctrl.AddProperty(prop);
}

void OutputLayerPropertyConfigure::PropertyChanged(CModelGridProperty* prop, AbstractLayer* layer, bool& reload)
{
	np::nsas::_LAYER_STRUCTURE_UNION& entry = ((HiddenLayer*)layer)->GetEntry();

	void* data = (void*)prop->GetData();
	if (data == &entry.output.loss_type)
	{
		CString value = prop->GetValue();
		for (neuro_u32 i = 0; i < _countof(loss_type_string); i++)
		{
			if (value == loss_type_string[i])
			{
				entry.output.loss_type = i;
				reload=true;
				break;
			}
		}
	}
}
