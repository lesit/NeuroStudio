#if !defined(_STUDIO_MENU_H)
#define _STUDIO_MENU_H

namespace np
{
	namespace studio
	{
		enum class _menu
		{
			split = 1000,
			link_del,
			model_del,
			layer_add,
			output_layer_add,
			layer_multi_del,
			bin_reader_add_to_input,
			text_reader_add_to_input,
			producer_add,
			erase_display,
			clear_all_displays,
			display_all_layers,
		};

		struct _menu_item
		{
			_menu_item(_menu id, neuro_u32 str_id = 0, bool disable=false, bool is_check=false)
			{
				this->id = id;
				this->str_id = str_id;
				this->disable = disable;
//				this->has_check = has_check;
				this->is_check = is_check;
			}
			_menu id;
			neuro_u32 str_id;

			bool disable;
//			bool has_check;
			bool is_check;
		};
	}
}

#endif
