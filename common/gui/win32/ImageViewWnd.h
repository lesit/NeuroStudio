#pragma once

#include "../../common.h"

namespace np
{
	namespace gui
	{
		namespace win32
		{
			class CImageViewWnd : public CWnd
			{
			public:
				CImageViewWnd(bool bCenter=true, bool bStretch=false);
				virtual ~CImageViewWnd();

				neuron_value* AllocBuffer(neuro_u32 width, neuro_u32 height);

			protected:
				DECLARE_MESSAGE_MAP()
				afx_msg void OnPaint();

				_VALUE_VECTOR m_imported_img_buffer;
				neuro_u32 m_width;
				neuro_u32 m_height;

				bool m_bCenter;
				bool m_bStretch;
			};
		}
	}
}