#if !defined(_IMAGE_PROCESSING_H)
#define _IMAGE_PROCESSING_H

namespace np
{
	namespace gui
	{
		struct _IMAGEDATA_MONO_SCALE_INFO
		{
			_IMAGEDATA_MONO_SCALE_INFO()
			{
				red_scale = 0.587f;
				green_scale = 0.299f;
				blue_scale = 0.114f;
			}

			neuro_float red_scale;
			neuro_float green_scale;
			neuro_float blue_scale;
		};
	}
}

#endif
