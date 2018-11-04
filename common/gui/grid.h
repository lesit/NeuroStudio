#pragma once

#include "shape.h"
namespace np
{
	namespace gui
	{
		struct _GRID_LAYOUT
		{
			_GRID_LAYOUT()
			{
				SetLayout({ 0, 0, 0, 0 }, { 100,50 }, { 50, 25 });
			}

			void SetLayout(const NP_RECT& view_margin, const NP_SIZE& item_size, const NP_SIZE& item_margin)
			{
				this->view_margin = view_margin;

				grid_size.width = 2 * item_margin.width + item_size.width;
				grid_size.height = 2 * item_margin.height + item_size.height;

				this->item_margin = item_margin;
				this->item_size = item_size;
			}

			NP_RECT view_margin;

			NP_SIZE grid_size;

			NP_SIZE item_margin;
			NP_SIZE item_size;
		};
	}
}
