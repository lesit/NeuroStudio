#include "stdafx.h"

#include "NetworkMatrix.h"

using namespace np;
using namespace np::network;
using namespace np::gui;

NetworkMatrix::NetworkMatrix()
{
	m_axis.Set(0, 0);
	m_max_row_count = 0;
}

void NetworkMatrix::SetGridLayout(const NP_RECT& view_margin, const NP_SIZE& item_size, const NP_SIZE& item_margin)
{
	m_grid_layout.SetLayout(view_margin, item_size, item_margin);
	MakeLayerLayouts();	// link의 line들을 다시 구해야 하므로
}

// 한번 load 되거나 삽입/삭제로 인해 matrix가 새로 구성되었을때 모든 layer 및 link 정보를 새로 갱신한다.
void NetworkMatrix::MakeLayerLayouts()
{
	m_max_row_count = 0;

	ClearAllLinks();
	for (neuro_u32 level = 0, last_level = m_matrix.size(); level < last_level; level++)
	{
		m_max_row_count = max(m_max_row_count, m_matrix[level].size());

		for (neuro_u32 row = 0, last_row = m_matrix[level].size(); row < last_row; row++)
		{
			AbstractLayer* layer = m_matrix[level][row];
			if (layer == NULL)
				continue;

			layer->gui_grid_point.x = level;
			layer->gui_grid_point.y = row;

			SetLayerInputLinkInfo(*layer);	// get all links
		}
	}
}

bool NetworkMatrix::LayerHitTest(const NP_POINT& pt, _POS_INFO_IN_LAYER& unit_info) const
{
	unit_info.pos_in_grid = _POS_INFO_IN_LAYER::_pos_in_grid::none;

	unit_info.matrix_pt.level = (pt.x - m_grid_layout.view_margin.left) / m_grid_layout.grid_size.width;
	unit_info.matrix_pt.row = (pt.y - m_grid_layout.view_margin.top) / m_grid_layout.grid_size.height;
	if (unit_info.matrix_pt.level >= m_matrix.size() || unit_info.matrix_pt.row >= m_matrix[unit_info.matrix_pt.level].size())
		return false;

	unit_info.pos_in_grid = _POS_INFO_IN_LAYER::_pos_in_grid::layer;
	unit_info.layer = GetLayer(unit_info.matrix_pt);
	if(unit_info.layer)
	{
		NP_2DSHAPE layer_shape = GetLayerRect(unit_info.matrix_pt);

		if (pt.x < layer_shape.pt.x)
			unit_info.pos_in_grid = _POS_INFO_IN_LAYER::_pos_in_grid::side_left;
		else if (pt.x > layer_shape.pt.x + layer_shape.sz.width)
			unit_info.pos_in_grid = _POS_INFO_IN_LAYER::_pos_in_grid::side_right;
		else if (pt.y < layer_shape.pt.y)
			unit_info.pos_in_grid = _POS_INFO_IN_LAYER::_pos_in_grid::side_up;
		else if (pt.y > layer_shape.pt.y + layer_shape.sz.height)
			unit_info.pos_in_grid = _POS_INFO_IN_LAYER::_pos_in_grid::side_down;
	}

	return true;
}

MATRIX_SCOPE NetworkMatrix::GetMatrixScope(const NP_RECT& rc, bool is_include_all) const
{
	NP_SIZE grid_size = m_grid_layout.grid_size;

	MATRIX_SCOPE matrix_scope;
	matrix_scope.first.level = (rc.left - m_grid_layout.view_margin.left) / grid_size.width;
	matrix_scope.first.row = (rc.top - m_grid_layout.view_margin.top) / grid_size.height;

	matrix_scope.second.level = (rc.right - m_grid_layout.view_margin.left) / grid_size.width + 1;
	matrix_scope.second.row = (rc.bottom - m_grid_layout.view_margin.top) / grid_size.height + 1;

	if (!is_include_all)
	{	// layer rect까지 포함하는지
		NP_RECT rcLayer = GetLayerRect(matrix_scope.first);
		if (rc.left > rcLayer.left)
			++matrix_scope.first.level;
		if (rc.top > rcLayer.top)
			++matrix_scope.first.row;

		rcLayer = GetLayerRect(matrix_scope.second);
		if (rc.right < rcLayer.right)
			--matrix_scope.second.level;
		if (rc.bottom < rcLayer.bottom)
			--matrix_scope.second.row;
	}
	return matrix_scope;
}

void NetworkMatrix::SetLayerInputLinkInfo(const AbstractLayer& layer)
{
	MATRIX_POINT to_mp = GetLayerMatrixPoint(layer);

	std::vector<const AbstractLayer*> item_vector;
	GetInputItemVector(layer, item_vector);
	for (neuro_u32 i = 0; i < item_vector.size(); i++)
		AddLinkInfo(item_vector[i]->uid, GetLayerMatrixPoint(*item_vector[i]), layer.uid, to_mp);
}

void NetworkMatrix::AddLinkInfo(neuro_u32 from_uid, const MATRIX_POINT& from_layer_mp, neuro_u32 to_uid, const MATRIX_POINT& to_layer_mp)
{
	_LINK_INFO link;
	link.from = from_layer_mp;
	link.to = to_layer_mp;
	link.line.arrow_type = _line_arrow_type::end;

	NP_2DSHAPE from_shape = GetLayerRect(link.from);
	NP_2DSHAPE to_shape = GetLayerRect(link.to);
	if (link.from.level == link.to.level)
	{
		link.line.draw_type = _line_draw_type::straight;

		link.line.bezier.start = { from_shape.pt.x + from_shape.sz.width / 2, from_shape.pt.y + from_shape.sz.height };
		link.line.bezier.points.push_back(_BEZIER_POINT({ to_shape.pt.x + to_shape.sz.width / 2, to_shape.pt.y }));
	}
	else
	{
		link.line.bezier.start = { from_shape.pt.x + from_shape.sz.width, from_shape.pt.y + from_shape.sz.height / 2 };
		NP_POINT ptTo(to_shape.pt.x, to_shape.pt.y + to_shape.sz.height / 2);

		if (link.from.row == link.to.row)
		{
			link.line.draw_type = _line_draw_type::straight;
			link.line.bezier.points.push_back(_BEZIER_POINT(ptTo));
		}
		else if (link.from.level + 1 == link.to.level)	// 바로 다음에 있을 때
		{
			link.line.draw_type = _line_draw_type::bezier;
			link.line.bezier.points.push_back(_BEZIER_POINT(ptTo, true, 1.f));
		}
		else// 두 단계 이상 떨어져 있을 때
		{
			link.line.draw_type = _line_draw_type::bezier;

			const bool is_down_link = link.from.row < link.to.row;
			neuro_u32 center_row = is_down_link ? link.to.row : link.from.row;

			bool is_empty_row = true;
			for (neuro_u32 level = link.from.level + 1; level < link.to.level; level++)
			{
				if (GetLayer(level, center_row) != NULL)
				{
					is_empty_row = false;
					break;
				}
			}

			neuro_float factor = is_down_link ? -1.f : 1.f;
			/*	선이 겹칠까 염려되었지만,
				보통 NNMatrixModify::MakeMatrix에서 겹치지 않게 layer의 위치를 배열하므로 그럴일이 없을 것이다.
			*/
			if (is_empty_row)	// 만약 to layer의 level에서 row가 비었으면 거기에서 선을 그으면 된다.
			{
				NP_POINT pt1;
				pt1.x = GetLayerX(link.from.level + 1);

				link.overlap.mp = { link.from.level + 1, link.to.row };

				_GRID_LINK_OVERLAP& overlap = m_grid_link_overlap_map[link.overlap.mp.value];
				link.overlap.index = overlap.FindSmallest(true, is_down_link);
				overlap.Update(true, is_down_link, link.overlap.index, true);

				pt1.y = ptTo.y + factor * (neuro_float(m_grid_layout.item_size.height) / 5.f) * (link.overlap.index + 1);

				link.line.bezier.points.push_back(_BEZIER_POINT(pt1, true, 1.f));
				link.line.bezier.points.push_back(_BEZIER_POINT(ptTo, false, 0.5f, false));
			}
			else // 그렇지 않으면, to layer의 row 바로 위에 중간 선을 그어야 한다. 이건 link 추가할때 발생. 보통 발생하지 않는다.
			{
				NP_POINT pt1, pt2;

				pt1.x = GetLayerX(link.from.level + 1);
				pt2.x = GetLayerX(link.to.level - 1) + m_grid_layout.item_size.width;

				link.overlap.mp = { link.from.level + 1, is_down_link ? center_row - 1 : center_row };

				_GRID_LINK_OVERLAP& overlap = m_grid_link_overlap_map[link.overlap.mp.value];
				link.overlap.index = overlap.FindSmallest(false, is_down_link);
				overlap.Update(false, is_down_link, link.overlap.index, true);

				pt1.y = GetGridY(link.overlap.mp.row) + factor * (neuro_float(m_grid_layout.item_margin.height) / 5.f) * (link.overlap.index + 1);
				if (is_down_link)
					pt1.y += m_grid_layout.grid_size.height;

				pt2.y = pt1.y;

				link.line.bezier.points.push_back(_BEZIER_POINT(pt1, true, 1.5f));
				link.line.bezier.points.push_back(_BEZIER_POINT(pt2, true, 1.f));
				link.line.bezier.points.push_back(_BEZIER_POINT(ptTo, true, 1.5f));
			}
		}
	}

	_u64_union u64_uid;
	u64_uid.upper = from_uid;
	u64_uid.lower = to_uid;
	m_link_map[u64_uid.u64] = link;
}

void NetworkMatrix::RemoveLinkInfo(neuro_u32 from_uid, neuro_u32 to_uid)
{
	_u64_union uid64;
	uid64.upper = from_uid;
	uid64.lower = to_uid;
	_link_map::iterator it = m_link_map.find(uid64.u64);
	if (it != m_link_map.end())
	{
		const _LINK_INFO& link = it->second;
		const bool is_down_link = link.from.row < link.to.row;
		if (link.line.bezier.points.size() >= 2)
		{
			_GRID_LINK_OVERLAP& overlap = m_grid_link_overlap_map[link.overlap.mp.value];
			overlap.Update(link.line.bezier.points.size()==3, is_down_link, link.overlap.index, false);
		}
		m_link_map.erase(it);
	}
}

void NetworkMatrix::ClearAllLinks()
{
	m_link_map.clear();
	m_grid_link_overlap_map.clear();
}

// 일괄적으로 부분 level들과 row를 삭제할 경우 RemoveLevel/RemoveRow를 매번 호출하는 것 보다 한번에 처리하는게 훨씬 낫다.
void NetworkMatrix::BatchMatrixRemove(const _std_u32_vector& level_index_vector, const _std_u32_vector& row_index_vector)
{
	if (level_index_vector.size() == 0 && row_index_vector.size() == 0)
		return;

	for (neuro_32 i = level_index_vector.size() - 1; i >= 0; i--)
	{
		if (level_index_vector[i] < m_matrix.size())
			m_matrix.erase(m_matrix.begin() + level_index_vector[i]);
	}

	for (neuro_32 level = 0; level <m_matrix.size(); level++)
	{
		_layer_vector& row_layer_vector = m_matrix[level];
		for (neuro_32 i = row_index_vector.size() - 1; i >= 0; i--)
		{
			if (row_index_vector[i] < row_layer_vector.size())
				row_layer_vector.erase(row_layer_vector.begin() + row_index_vector[i]);
		}
	}
	MakeLayerLayouts();
}

bool NetworkMatrix::IsLevelEmpty(neuro_u32 level) const
{
	if (level >= m_matrix.size())
		return true;

	const _layer_vector& layer_vector = m_matrix[level];
	for (neuro_u32 row = 0; row < layer_vector.size(); row++)
	{
		if (layer_vector[row] != NULL)
			return false;
	}
	return true;
}

void NetworkMatrix::InsertLevel(neuro_u32 level, bool update_layer_pt)
{
	ModifyLevel(_modify::insert, level, update_layer_pt);
}

void NetworkMatrix::RemoveLevel(neuro_u32 level, bool update_layer_pt)
{
	ModifyLevel(_modify::remove, level, update_layer_pt);

	CalcMaxRowCount();	// 삭제되면 row를 다시 계산
}

// 하나씩 삽입/삭제할 경우 level의 위치에 따라 왼쪽 또는 오른쪽의 layer들만 위치 변경하면 된다.
inline void NetworkMatrix::ModifyLevel(_modify modify, neuro_u32 modify_level, bool update_layer_pt)
{
	if (update_layer_pt)
	{
		if (modify_level <m_matrix.size() / 2)
		{
			neuro_32 dec = modify == _modify::insert ? -1 : 1;

			// 왼쪽만 변경할 경우 좌표축을 이동시킨다.
			m_axis.x += modify == _modify::insert ? 1 : -1;
			for (neuro_32 level = modify_level - 1; level >=0; level--)
			{
				_layer_vector& row_layer_vector = m_matrix[level];
				for (neuro_32 row = row_layer_vector.size() - 1; row >= 0; row--)
				{
					AbstractLayer* layer = (AbstractLayer*)row_layer_vector[row];
					if (layer == NULL)
						continue;

					layer->gui_grid_point.x += dec;

					// 출력 link 만 업데이트 해주면 된다.
					MATRIX_POINT from_layer_mp = GetLayerMatrixPoint(*layer);
					
					std::vector<const AbstractLayer*> item_vector;
					GetOutputItemVector(*layer, item_vector);
					for (neuro_u32 i=0;i<item_vector.size();i++)
						AddLinkInfo(layer->uid, from_layer_mp, item_vector[i]->uid, GetLayerMatrixPoint(*item_vector[i]));
				}
			}
		}
		else
		{	// 오른쪽을 변경할 경우 좌표축을 이동시킬 필요는 없다.
			neuro_32 dec = modify == _modify::insert ? 1 : -1;
			neuro_u32 level = modify == _modify::insert ? modify_level : modify_level + 1;
			neuro_u32 last_level = m_matrix.size();
			for (; level<last_level; level++)
			{
				_layer_vector& row_layer_vector = m_matrix[level];
				for (neuro_u32 row = 0; row < row_layer_vector.size(); row++)
				{
					AbstractLayer* layer = (AbstractLayer*)row_layer_vector[row];
					if (layer == NULL)
						continue;

					layer->gui_grid_point.x += dec;

					// 입력 link 만 업데이트 해주면 된다.
					SetLayerInputLinkInfo(*layer);
				}
			}
		}
	}

	if (modify == _modify::insert)
		m_matrix.insert(m_matrix.begin() + modify_level, _layer_vector());
	else
		m_matrix.erase(m_matrix.begin() + modify_level);
}

bool NetworkMatrix::IsRowEmpty(neuro_u32 row) const
{
	for (neuro_u32 level = 0, n = m_matrix.size(); level < n; level++)
	{
		const _layer_vector& row_layer_vector = m_matrix[level];
		if (row >= row_layer_vector.size())
			continue;

		if (row_layer_vector[row] != NULL)
			return false;
	}
	return true;
}

void NetworkMatrix::InsertRow(neuro_u32 row, bool update_layer_pt)
{
	ModifyRow(_modify::insert, row, update_layer_pt);
}

void NetworkMatrix::RemoveRow(neuro_u32 row, bool update_layer_pt)
{
	ModifyRow(_modify::remove, row, update_layer_pt);
}

void NetworkMatrix::ModifyRow(_modify modify, neuro_u32 modify_row, bool update_layer_pt)
{
	neuro_u32 row_size = GetMaxRowCount();
	if (row_size == 0)
		return;

	neuro_u32 start_row, last_row;
	neuro_32 dec;
	if (modify_row < row_size / 2)
	{
		start_row = 0; last_row = modify_row;
		dec = -1;

		if(update_layer_pt)
			m_axis.y += modify == _modify::insert ? 1 : -1;
	}
	else
	{
		start_row = modify == _modify::insert ? modify_row : modify_row + 1;
		last_row = row_size;
		dec = 1;
	}
	if (modify == _modify::remove)
		dec = -dec;

	m_max_row_count = 0;
	for (neuro_u32 level = 0, n = m_matrix.size(); level < n; level++)
	{
		_layer_vector& row_layer_vector = m_matrix[level];
		m_max_row_count = max(m_max_row_count, row_layer_vector.size());

		if (modify_row >= row_layer_vector.size())
			continue;

		if (update_layer_pt)
		{
			for (neuro_u32 row = start_row, last = min(row_layer_vector.size(), last_row); row < last; row++)
			{
				AbstractLayer* layer = (AbstractLayer*)row_layer_vector[row];
				if (layer == NULL)
					continue;

				layer->gui_grid_point.y += dec;

				// 입력 link 만 업데이트 해주면 된다.
				SetLayerInputLinkInfo(*layer);
			}
		}
		if(modify==_modify::insert)
			row_layer_vector.insert(row_layer_vector.begin() + modify_row, NULL);
		else
			row_layer_vector.erase(row_layer_vector.begin() + modify_row);
	}
}

void NetworkMatrix::MoveRows(const _MOVING_SCOPE& moving_scope, bool update_layer_pt, neuro_32 move_count)
{
	if (move_count == 0)
		return;

	if (moving_scope.moving_level_scope_vector.size() == 0)
		return;

	auto available_move_rows = [&](neuro_u32 start_level, neuro_u32 last_level, neuro_32 start_row)
	{
		if (start_row + move_count < 0)
			return false;

		// 위로 올릴땐 비어 있는 곳인지 검사해야 한다.
		for (neuro_u32 level = start_level; level < last_level; level++)
		{
			_layer_vector& row_layer_vector = m_matrix[level];
			if (row_layer_vector[start_row] == NULL) continue;

			if (row_layer_vector[start_row + move_count] != NULL)	// 더이상 올릴수 없다
				return false;
		}
		return true;
	};

	auto move_rows = [&](neuro_u32 start_level, neuro_u32 last_level, neuro_32 start_row)
	{
		for (neuro_u32 level = start_level; level < last_level; level++)
		{
			_layer_vector& row_layer_vector = m_matrix[level];
			if (update_layer_pt)
			{
				for (neuro_32 row = start_row; row < row_layer_vector.size(); row++)
				{
					if (row_layer_vector[row] != NULL)
						row_layer_vector[row]->gui_grid_point.y += move_count;
				}
			}
			if (move_count > 0)
				row_layer_vector.insert(row_layer_vector.begin() + start_row, move_count, NULL);
			else
				row_layer_vector.erase(row_layer_vector.begin() + start_row + move_count, row_layer_vector.begin() + start_row);
		}
	};

	std::pair<neuro_u32, neuro_u32> prev_scope;
	prev_scope.first = prev_scope.second = moving_scope.moving_level_scope_vector[0].second;

	// 밀어낼땐 level 별로 이동존의 row이하 모두 밀어낼수 있다.
	neuro_u32 row = moving_scope.start_row;
	for (neuro_u32 i = 0, n = moving_scope.moving_level_scope_vector.size(); i < n; i++, row++)
	{
		std::pair<neuro_u32, neuro_u32> scope = moving_scope.moving_level_scope_vector[i];

		if (move_count < 0 && 
			(!available_move_rows(scope.first, prev_scope.first, row)
			|| !available_move_rows(prev_scope.second, scope.second, row)))
			break;

		move_rows(scope.first, prev_scope.first, row);
		move_rows(prev_scope.second, scope.second, row);

		prev_scope = scope;
	}
}
