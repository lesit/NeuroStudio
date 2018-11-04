#include "stdafx.h"
#include "DataProducerPropertyConfigure.h"

#include "NeuroData/model/ImageFileProducerModel.h"

using namespace property;

ImageProducerPropertyConfigure::ImageProducerPropertyConfigure()
{
}

void ImageProducerPropertyConfigure::CompositeProperties(CMFCPropertyGridCtrl& list_ctrl, AbstractProducerModel* model)
{
	ImageFileProducerModel* producer = (ImageFileProducerModel*)model;

	const _NEURO_INPUT_IMAGEFILE_INFO& info = producer->GetDefinition();

	{
		CModelGridProperty* color_prop = new CModelGridProperty(L"Color");
		list_ctrl.AddProperty(color_prop);

		CModelGridProperty* color_scale_prop = new CModelGridProperty(L"Scale type"
			, (_variant_t)ToString(info.color_type)
			, L"scale fixel's color to mono or use all color"
			, (DWORD_PTR)_prop_type::color_type);
		for (neuro_u32 i = 0; i < _countof(_color_type_string); i++)
			color_scale_prop->AddOption(_color_type_string[i]);
		color_scale_prop->AllowEdit(FALSE);
		color_prop->AddSubItem(color_scale_prop);

		if (info.color_type == _color_type::mono)
		{
			CModelGridProperty* prop = new CModelGridProperty(L"Red", (_variant_t)info.mono_scale.red_scale
				, L"Red color scale factor", (DWORD_PTR)_prop_type::red_scale);
			color_prop->AddSubItem(prop);

			prop = new CModelGridProperty(L"Green", (_variant_t)info.mono_scale.green_scale
				, L"Red color scale factor", (DWORD_PTR)_prop_type::green_scale);
			color_prop->AddSubItem(prop);

			prop = new CModelGridProperty(L"Blue", (_variant_t)info.mono_scale.blue_scale
				, L"Red color scale factor", (DWORD_PTR)_prop_type::blue_scale);
			color_prop->AddSubItem(prop);
		}
	}

	{
		CModelGridProperty* resolution_prop = new CModelGridProperty(L"Resolution");
		list_ctrl.AddProperty(resolution_prop);

		CModelGridProperty* prop = new CModelGridProperty(L"Width", (_variant_t)info.sz.width
			, L"Output width", (DWORD_PTR)_prop_type::width);
		resolution_prop->AddSubItem(prop);

		prop = new CModelGridProperty(L"Height", (_variant_t)info.sz.height
			, L"Output height", (DWORD_PTR)_prop_type::height);
		resolution_prop->AddSubItem(prop);

		prop = new CModelGridProperty(L"Fit size type"
			, (_variant_t)ToString(info.fit_type)
			, L"Type to fit image size"
			, (DWORD_PTR)_prop_type::fit_type);
		for (neuro_u32 i = 0; i < _countof(stretch_type_string); i++)
			prop->AddOption(stretch_type_string[i]);
		prop->AllowEdit(FALSE);

		resolution_prop->AddSubItem(prop);
	}
}

void ImageProducerPropertyConfigure::PropertyChanged(CModelGridProperty* prop, AbstractProducerModel* model, bool& reload) const
{
	ImageFileProducerModel* producer = (ImageFileProducerModel*)model;

	_NEURO_INPUT_IMAGEFILE_INFO info = producer->GetDefinition();
	switch ((_prop_type)prop->GetData())
	{
	case _prop_type::color_type:
	{
		CString value = prop->GetValue();
		for (neuro_u32 i = 0; i < _countof(_color_type_string); i++)
		{
			if (value == _color_type_string[i])
			{
				info.color_type = (_color_type)i;
				reload = true;
				break;
			}
		}
		break;
	}
	case _prop_type::red_scale:
		info.mono_scale.red_scale = prop->GetValue().fltVal;
		break;
	case _prop_type::green_scale:
		info.mono_scale.green_scale = prop->GetValue().fltVal;
		break;
	case _prop_type::blue_scale:
		info.mono_scale.blue_scale = prop->GetValue().fltVal;
		break;
	case _prop_type::width:
		info.sz.width = prop->GetValue().intVal;
		reload = true;
		break;
	case _prop_type::height:
		info.sz.height = prop->GetValue().intVal;
		reload = true;
		break;
	case _prop_type::fit_type:
	{
		CString value = prop->GetValue();
		for (neuro_u32 i = 0; i < _countof(stretch_type_string); i++)
		{
			if (value == stretch_type_string[i])
			{
				info.fit_type = (_stretch_type)i;
				break;
			}
		}
		break;
	}
	}
	producer->SetDefinition(info);
}
