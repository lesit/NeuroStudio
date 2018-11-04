#pragma once

#include "common.h"
#include "gui/shape.h"

struct _LABEL_INFO
{
	_LABEL_INFO(long height = 15, long width = 5, long lfWeight = FW_REGULAR, COLORREF color=RGB(0, 0, 255))
	{
		memset(&logFont, 0, sizeof(LOGFONT));
		logFont.lfHeight = height;
		logFont.lfWidth = width;
		logFont.lfWeight = lfWeight;

		this->color = color;
	}
	LOGFONT logFont;
	COLORREF color;
};

struct _DRAW_LAYER_INFO
{
	_LABEL_INFO layer_label;
	NP_SIZE draw_size;

	_DRAW_LAYER_INFO(neuro_u32 width, neuro_u32 height)
	{
		draw_size.width = width;
		draw_size.height = height;
	}

	_DRAW_LAYER_INFO(const NP_SIZE& draw_size = NP_SIZE(150, 150))
	{
		this->draw_size = draw_size;
	}
};
