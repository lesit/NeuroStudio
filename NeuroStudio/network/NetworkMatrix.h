#pragma once

#include "NeuroKernel/network/HiddenLayer.h"
#include "gui/line.h"
#include "gui/grid.h"
#include "util/StringUtil.h"

namespace np
{
	namespace network
	{
		union MATRIX_POINT
		{
			MATRIX_POINT()
			{
				value = neuro_last64;
			}
			MATRIX_POINT(neuro_u32 _level, neuro_u32 _row)
			{
				level = _level;
				row = _row;
			}

			void Init()
			{
				value = neuro_last64;
			}
			bool IsValid() const
			{
				return value != neuro_last64;
			}
			bool operator == (const MATRIX_POINT& src) const
			{
				return value == src.value;
			}

			bool operator != (const MATRIX_POINT& src) const
			{
				return value != src.value;
			}

			std::wstring ToString() const
			{
				return util::StringUtil::Format<wchar_t>(L"Level:%u, Row:%u", level, row);
			}
			std::wstring ToSimpleString() const
			{
				return util::StringUtil::Format<wchar_t>(L"%u, %u", level, row);
			}

			struct
			{
				neuro_u32 row;
				neuro_u32 level;
			};
			neuro_u64 value;
		};

		struct MATRIX_SCOPE
		{
			void Initialize()
			{
				first.Init();
				second.Init();
			}

			bool IsValid() const
			{
				return first.level<second.level && first.row<second.row;
			}

			bool InScope(const MATRIX_POINT& pt) const
			{
				return InScope(pt.level, pt.row);
			}
			bool InScope(neuro_u32 level, neuro_u32 row) const
			{
				return level >= first.level && level < second.level && row >= first.row && row < second.row;
			}

			MATRIX_POINT first;
			MATRIX_POINT second;
		};

		struct _POS_INFO_IN_LAYER
		{
			void Initialize()
			{
				matrix_pt.Init();
				pos_in_grid = _pos_in_grid::none;
				layer = NULL;
			}
			MATRIX_POINT matrix_pt;
			AbstractLayer* layer;

			enum class _pos_in_grid { none, layer, side_left, side_right, side_up, side_down };
			_pos_in_grid pos_in_grid;
		};

		struct _GRID_LINK_OVERLAP
		{
			_GRID_LINK_OVERLAP()
			{
				memset(this, 0, sizeof(_GRID_LINK_OVERLAP));
			}

			struct _OVERLAP
			{
				neuro_32 up_links[4];
				neuro_32 down_links[4];
			};
			_OVERLAP item_margin;
			_OVERLAP center;

			neuro_32* GetOverlap(bool is_center, bool is_down)
			{
				_OVERLAP& overlap = is_center ? center : item_margin;
				return is_down ? overlap.down_links : overlap.up_links;
			}
			void Update(bool is_center, bool is_down, neuro_u32 index, bool is_increase)
			{
				neuro_32* overlap = GetOverlap(is_center, is_down);
				is_increase ? ++overlap[index]: --overlap[index];
			}
			neuro_u32 FindSmallest(bool is_center, bool is_down)
			{
				neuro_32* overlap = GetOverlap(is_center, is_down);
				neuro_32 n = overlap[0];
				for (neuro_u32 i = 1; i < 4; i++)
				{
					if (overlap[i] < n)
						return i;
				}
				return 0;
			}
		};
		typedef std::unordered_map<neuro_u64, _GRID_LINK_OVERLAP> _grid_link_overlap_map;

		struct _LINK_INFO
		{
			~_LINK_INFO() {}

			bool HasLink() const { return from.IsValid() && to.IsValid(); }
			bool operator==(const _LINK_INFO& src) const { return from == src.from && to == src.to; }
			bool operator!=(const _LINK_INFO& src) const { return from != src.from || to != src.to; }

			MATRIX_POINT from, to;

			gui::_CURVE_INTEGRATED_LINE line;

			struct _OVERLAP
			{
				MATRIX_POINT mp;
				neuro_u32 index;
			};
			_OVERLAP overlap;
		};

		typedef std::unordered_map<neuro_u64, _LINK_INFO> _link_map;

		class NetworkMatrix
		{
		public:
			NetworkMatrix();
			virtual ~NetworkMatrix() {}

			void SetGridLayout(const NP_RECT& view_margin, const NP_SIZE& item_size, const NP_SIZE& item_margin);
			const gui::_GRID_LAYOUT& GetGridLayout() const { return m_grid_layout; }

			neuro_u32 GetMatrixWidth() const
			{
				return m_grid_layout.view_margin.left + GetLevelCount() * m_grid_layout.grid_size.width + m_grid_layout.view_margin.right;
			}

			neuro_u32 GetMatrixHeight() const
			{
				return m_grid_layout.view_margin.top + GetMaxRowCount() * m_grid_layout.grid_size.height + m_grid_layout.view_margin.bottom;
			}

			inline neuro_u32 GetGridX(neuro_32 level) const
			{
				return m_grid_layout.view_margin.left + level * m_grid_layout.grid_size.width;
			}

			inline neuro_u32 GetGridY(neuro_32 row) const
			{
				return m_grid_layout.view_margin.top + row * m_grid_layout.grid_size.height;
			}

			inline NP_RECT GetGridRect(const MATRIX_POINT& pt) const
			{
				NP_RECT ret;
				ret.left = GetGridX(pt.level);
				ret.right = ret.left + m_grid_layout.grid_size.width;
				ret.top = GetGridY(pt.row);
				ret.bottom = ret.top + m_grid_layout.grid_size.height;
				return ret;
			}

			inline neuro_u32 GetLayerX(neuro_32 level) const
			{
				return GetGridX(level) + m_grid_layout.item_margin.width;
			}

			inline neuro_u32 GetLayerY(neuro_32 row) const
			{
				return GetGridY(row) + m_grid_layout.item_margin.height;
			}

			inline NP_RECT GetLayerRect(const MATRIX_POINT& pt) const
			{
				NP_RECT ret;
				ret.left = GetLayerX(pt.level);
				ret.right = ret.left + m_grid_layout.item_size.width;
				ret.top = GetLayerY(pt.row);
				ret.bottom = ret.top + m_grid_layout.item_size.height;
				return ret;
			}

			void ResetMatrix()
			{
				m_matrix.clear();
				m_max_row_count = 0;

				m_axis = { 0,0 };
				m_grid_link_overlap_map.clear();

				m_link_map.clear();
			}

			inline neuro_u32 GetLevelCount() const { return m_matrix.size(); }
			inline neuro_u32 GetRowCount(neuro_u32 level) const { return level >= m_matrix.size() ? 0 : m_matrix[level].size(); }
			inline neuro_u32 GetMaxRowCount() const { return m_max_row_count;}
			inline neuro_u32 CalcMaxRowCount()
			{
				m_max_row_count = 0;
				for (neuro_u32 level = 0; level < m_matrix.size(); level++)
					m_max_row_count = max(m_max_row_count, m_matrix[level].size());
				return m_max_row_count;
			}
			inline AbstractLayer* GetLayer(neuro_u32 level, neuro_u32 row) const
			{
				if (level < 0 || level >= m_matrix.size() || row < 0 || row >= m_matrix[level].size())
					return NULL;
				return m_matrix[level][row];
			}
			inline AbstractLayer* GetLayer(const MATRIX_POINT& pt) const
			{
				return GetLayer(pt.level, pt.row);
			}

			inline void SetLayer(neuro_u32 level, neuro_u32 row, AbstractLayer* layer)
			{
				if (layer == NULL)
				{
					if (level >= m_matrix.size() || row >= m_matrix[level].size())
						return;
					m_matrix[level][row] = NULL;
				}
				else
				{
					if (level >= m_matrix.size())
						m_matrix.resize(level + 1);
					
					if (row >= m_matrix[level].size())
						m_matrix[level].resize(row + 1, NULL);

					m_matrix[level][row] = layer;
					layer->gui_grid_point.x = (neuro_32)level - m_axis.x;
					layer->gui_grid_point.y = (neuro_32)row - m_axis.y;
				}
			}
			inline void SetLayer(const MATRIX_POINT& mp, AbstractLayer* layer)
			{
				SetLayer(mp.level, mp.row, layer);
			}

			inline MATRIX_POINT GetLayerMatrixPoint(const AbstractLayer& layer) const
			{
				if (layer.gui_grid_point.x + m_axis.x < 0 || layer.gui_grid_point.y + m_axis.y < 0)
					return MATRIX_POINT();
				return{ (neuro_u32)(layer.gui_grid_point.x + m_axis.x), (neuro_u32)(layer.gui_grid_point.y + m_axis.y) };
			}

			bool LayerHitTest(const NP_POINT& pt, _POS_INFO_IN_LAYER& unit_info) const;

			MATRIX_SCOPE GetMatrixScope(const NP_RECT& rc, bool is_include_all) const;

			const _link_map& GetLinkMap() const { return m_link_map; }
		protected:
			virtual neuro_u32 GetItemUid(const AbstractLayer& item) const = 0;
			virtual void GetInputItemVector(const AbstractLayer& item, std::vector<const AbstractLayer*>& item_vector) const = 0;
			virtual void GetOutputItemVector(const AbstractLayer& item, std::vector<const AbstractLayer*>& item_vector) const = 0;

			void MakeLayerLayouts();
			void SetLayerInputLinkInfo(const AbstractLayer& layer);
			void AddLinkInfo(neuro_u32 from_uid, const MATRIX_POINT& from_layer_mp, neuro_u32 to_uid, const MATRIX_POINT& to_layer_mp);
			void RemoveLinkInfo(neuro_u32 from_uid, neuro_u32 to_uid);
			void ClearAllLinks();

			void BatchMatrixRemove(const _std_u32_vector& level_index_vector, const _std_u32_vector& row_index_vector);

			bool IsLevelEmpty(neuro_u32 level) const;
			void InsertLevel(neuro_u32 level, bool update_layer_pt);
			void RemoveLevel(neuro_u32 level, bool update_layer_pt);

			bool IsRowEmpty(neuro_u32 row) const;
			void InsertRow(neuro_u32 row, bool update_layer_pt);
			void RemoveRow(neuro_u32 row, bool update_layer_pt);

			struct _MOVING_SCOPE
			{
				neuro_32 start_row;

				std::vector<std::pair<neuro_u32, neuro_u32>> moving_level_scope_vector;
			};
			void MoveRows(const _MOVING_SCOPE& moving_scope, bool update_layer_pt, neuro_32 move_count = 1);

		protected:
			enum class _modify{insert, remove};
			void ModifyLevel(_modify modify, neuro_u32 level, bool update_layer_pt);
			void ModifyRow(_modify modify, neuro_u32 row, bool update_layer_pt);

			inline MATRIX_POINT GetRealMatrix(neuro_32 level, neuro_32 row) const
			{
				return{ (neuro_u32)(level + m_axis.x), (neuro_u32)(row + m_axis.y) };
			}

		private:
			gui::_GRID_LAYOUT m_grid_layout;

			NP_POINT m_axis;

			std::vector<_layer_vector> m_matrix;
			neuro_u32 m_max_row_count;

			_grid_link_overlap_map m_grid_link_overlap_map;
			_link_map m_link_map;
		};
	}
}
