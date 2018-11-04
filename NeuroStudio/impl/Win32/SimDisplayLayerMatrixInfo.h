#pragma once

#include "NeuroKernel/engine/NeuralNetworkProcessor.h"
#include "network/NetworkMatrix.h"

using namespace project;

class LayerDisplaySetup : public NeuroBindingModel
{
public:
	LayerDisplaySetup()
	{
		layer = NULL;
		memset(&display, 0, sizeof(_LAYER_DISPLAY_INFO));
	}

	virtual ~LayerDisplaySetup() {}

	LayerDisplaySetup& operator = (const LayerDisplaySetup& src)
	{
		mp = src.mp;
		layer = src.layer;
		display = src.display;
		return *this;
	}
	MATRIX_POINT mp;
	const AbstractLayer* layer;
	_LAYER_DISPLAY_INFO display;
};

typedef std::vector<LayerDisplaySetup> _layer_display_setup_row_vector;
typedef std::vector<_layer_display_setup_row_vector> _layer_display_setup_matrix_vector;

struct _LAYER_OUT_BUF
{
	neuron_value low_scale;
	neuron_value up_scale;

	_NEURO_TENSOR_DATA output;
	_TYPED_TENSOR_DATA<void*, 4> target;	// label이면 그냥 원래의 값을 넣도록 하자!
};

typedef std::unordered_map<neuro_u64, _LAYER_OUT_BUF> _layer_out_buf_map;

struct LayerDisplayItem
{
	MATRIX_POINT mp;
	neuro_u32 layer_uid;
	const engine::layers::AbstractLayerEngine* engine;

	_LAYER_DISPLAY_INFO display;
	_LAYER_OUT_BUF buffer;
};

typedef std::vector<LayerDisplayItem> _layer_display_item_rowl_vector;
typedef std::vector<_layer_display_item_rowl_vector> _layer_display_item_matrix_vector;
