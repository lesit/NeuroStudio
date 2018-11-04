#include "stdafx.h"

#include "AbstractLayer.h"
#include "HiddenLayer.h"

using namespace np;
using namespace np::network;


AbstractLayer::AbstractLayer(neuro_u32 _uid)
: uid(_uid)
{
	m_prev = m_next = NULL;
}

AbstractLayer::~AbstractLayer()
{
}

void AbstractLayer::RegisterOutput(HiddenLayer* layer)
{
	m_output_set.insert(layer);
}

void AbstractLayer::ReleaseOutput(HiddenLayer* layer)
{
	m_output_set.erase(layer);
}

void AbstractLayer::OnRemove()
{
	for (_hiddenlayer_set::iterator it = m_output_set.begin(); it != m_output_set.end(); it++)
		(*it)->ReleaseInput(this);
}

void AbstractLayer::CheckOutputTensor()
{
	for (_hiddenlayer_set::iterator it = m_output_set.begin(); it != m_output_set.end(); it++)
	{
		HiddenLayer* out_layer = *it;;
		out_layer->CheckOutputTensor();
	}
}
