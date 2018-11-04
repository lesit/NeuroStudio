#pragma once

#include "common.h"

#include "NeuroData/model/AbstractPreprocessorModel.h"
#include "network/NNMatrixModify.h"

static const wchar_t* szDataModelClipboardFormat = L"DataModelClipboardFormat";
static const neuro_u32 data_model_cf = RegisterClipboardFormat(szDataModelClipboardFormat);
struct _DATA_MODEL_DRAG_SOURCE
{
	dp::model::AbstractPreprocessorModel* model;

	static neuro_u32 cf;
};

static const wchar_t* szLayerClipboardFormat = L"NeuroLayerClipboardFormat";
static const neuro_u32 neuro_layer_cf = RegisterClipboardFormat(szLayerClipboardFormat);
struct _LAYER_DRAG_SOURCE
{
	MATRIX_POINT mp;

	static neuro_u32 cf;
};
