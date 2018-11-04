#include "stdafx.h"

#include "NNMatrixModify.h"

#include "util/np_util.h"

#include "NeuroKernel/network/HiddenLayer.h"
#include "NeuroKernel/network/HiddenLayer.h"

using namespace np::network;

NNMatrixModify::NNMatrixModify()
{
}

NNMatrixModify::~NNMatrixModify()
{
}

neuro_u32 NNMatrixModify::GetItemUid(const AbstractLayer& item) const
{
	return item.uid;
}

void NNMatrixModify::GetInputItemVector(const AbstractLayer& item, std::vector<const AbstractLayer*>& item_vector) const
{
	if (item.GetLayerType() == network::_layer_type::input
		|| ((HiddenLayer&)item).GetMainInput() == 0)
		return;

	const _slice_input_vector& input_vector = ((const HiddenLayer&)item).GetInputVector();
	for (neuro_u32 i = 0; i < input_vector.size(); i++)
		item_vector.push_back(input_vector[i].layer);

	const HiddenLayer* side_input = ((const HiddenLayer&)item).GetSideInput();
	if (side_input)
		item_vector.push_back(side_input);
}

void NNMatrixModify::GetOutputItemVector(const AbstractLayer& item, std::vector<const AbstractLayer*>& item_vector) const
{
	const _hiddenlayer_set& output_set = item.GetOutputSet();
	_hiddenlayer_set::const_iterator it = output_set.begin();
	for (; it != output_set.end(); it++)
		item_vector.push_back(*it);
}

void NNMatrixModify::NetworkChanged(NeuralNetwork* network)
{
	ResetMatrix();
	if (network)
		MakeMatrix(network->GetInputLayers(), network->GetHiddenLayers());

	MakeLayerLayouts();
}

// 입력이 없는 layer의 경우 정확한 위치가 없기 때문에 가상의 위치를 설정해준다.
void NNMatrixModify::UpdateNetwork(NeuralNetwork& network)
{
	HiddenLayer* layer = (HiddenLayer*)network.GetHiddenLayers().start;
	if (layer == NULL)
		return;

	neuro_u32 prev_layer_level = 0;
	MATRIX_POINT cur_mp = GetLayerMatrixPoint(*layer);
	while (layer)
	{
		HiddenLayer* next_layer = (HiddenLayer*)layer->GetNext();

		MATRIX_POINT next_mp;
		if(next_layer)
			next_mp= GetLayerMatrixPoint(*next_layer);
		if (layer->GetMainInput() == NULL)	// 입력이 없을 경우 가상 입력을 만든다.
		{
			nsas::_VIRTUAL_POSITION vp;

			if (prev_layer_level < cur_mp.level)
			{
				if(next_layer && cur_mp.level == next_mp.level)
					vp.level_status = 2;
				else
					vp.level_status = 3;
			}
			else
			{
				if (next_layer && next_mp.level > cur_mp.level)	// 다음 layer는 다음 level에서 시작한다.
					vp.level_status = 1;
				else
					vp.level_status = 0;
			}

			for (neuro_32 row = cur_mp.row; row >= 0; row--)
			{
				AbstractLayer* vp_input = GetLayer(cur_mp.level - 1, cur_mp.row);
				if (vp_input)
				{
					vp.virtual_input = vp_input->uid;
					break;
				}
			}
			layer->SetVirtualPosition(vp);
		}
		prev_layer_level = cur_mp.level;
		cur_mp = next_mp;
		layer = next_layer;
	}
}

// loading, 새로고침 했을때만 한다.
void NNMatrixModify::MakeMatrix(const _LINKED_LAYER& input, const _LINKED_LAYER& hidden)
{
	MATRIX_POINT layer_pt(0, 0);

	std::unordered_map<neuro_u32, AbstractLayer*> uid_layer_map;

	AbstractLayer* input_layer = input.start;
	while (input_layer)
	{
		uid_layer_map[input_layer->uid] = input_layer;

		SetLayer(layer_pt, input_layer);

		++layer_pt.row;

		input_layer = input_layer->GetNext();
	}

	std::unordered_set<AbstractLayer*> main_connection_map;

	layer_pt.level = 1;
	layer_pt.row = 0;

	HiddenLayer* hidden_layer = (HiddenLayer*)hidden.start;
	while (hidden_layer)
	{
		uid_layer_map[hidden_layer->uid] = hidden_layer;

		AbstractLayer* first_in = GetFirstInputLayer(hidden_layer);
		if (first_in == NULL)
		{
			const nsas::_VIRTUAL_POSITION& vp = hidden_layer->GetVirtualPosition();

			if (vp.level_status == 2 || vp.level_status==3)	// 다음 level에서 시작
				++layer_pt.level;

			std::unordered_map<neuro_u32, AbstractLayer*>::const_iterator v_in_it = uid_layer_map.find(vp.virtual_input);
			if (v_in_it != uid_layer_map.end())
			{
				MATRIX_POINT in_mp = GetLayerMatrixPoint(*v_in_it->second);
				if (layer_pt.row < in_mp.row)
					layer_pt.row = in_mp.row + 1;
			}
			SetLayer(layer_pt, hidden_layer);
			if (vp.level_status == 1 || vp.level_status == 3)	// 다음 layer는 다음 level에서 시작한다.
			{
				++layer_pt.level;
				layer_pt.row = 0;
			}
			else
				++layer_pt.row;
		}
		else
		{
			// 입력 layer가 main connection이 등록 안 되었을 때 이 layer가 main connection 이다.
			bool is_main_connection = main_connection_map.find(first_in) == main_connection_map.end();
			if(is_main_connection)
				main_connection_map.insert(first_in);

			const _slice_input_vector& in_vector = hidden_layer->GetInputVector();
			for (neuro_u32 i = 0; i < in_vector.size(); i++)
			{
				MATRIX_POINT in_pt = GetLayerMatrixPoint(*in_vector[i].layer);
				if (in_pt.level == layer_pt.level)	// 입력과 같은 level이면 level 추가 필요
				{
					++layer_pt.level;
					layer_pt.row = 0;
					break;
				}
			}

			MATRIX_POINT first_in_pt = GetLayerMatrixPoint(*first_in);
			if (first_in_pt.row >= layer_pt.row) // 만약 첫번째 입력 layer의 위치보다 위에 있거나 같다면
			{
				// layer가 첫번째 입력 layer row위치와 같거나 그 아래에 있도록 하자
				layer_pt.row = first_in_pt.row;

				// 같은 row 위치인데 만약 2 level 이상 떨어져 있고 중간에 비어 있지 않다면 그 아래에 놓여야 한다. 10 p
				if (first_in_pt.level + 1 < layer_pt.level)
				{
					for (neuro_u32 level = first_in_pt.level + 1; level < layer_pt.level; level++)
					{
						// 만약 중간에 다른 layer가 있다면
						if (GetLayer(level, layer_pt.row) != NULL)	
						{
							++layer_pt.row;	// 한단계 아래에 위치하도록 하자
							break;
						}
					}
				}
			}
			else if (is_main_connection && first_in_pt.row < layer_pt.row)
			{	// 메인 연결인데 첫번째 입력 layer보다 아래에 있다면 입력을 내려버린다
				PushDownLayer(first_in_pt, true, layer_pt.level, layer_pt.row - first_in_pt.row);
			}
			/*	item_size = { 100,50 }, margin_in_grid = { 50, 25 } 으로 했을때 바로 앞에 layer가 있어도 이상하지 않음
			하지만 그 layer 들은 다른 출력을 가질 경우가 많기 때문에 독립성을 보여주기 위해 그냥 이대로 출력한다.
			*/
			// 같은 row위치에서 앞에 또 다른 layer가 있다면 그 layer를 아래로 밀어버리자
			MATRIX_POINT prev_mp(layer_pt.level - 1, layer_pt.row);
			if (layer_pt.row != first_in_pt.row && GetLayer(prev_mp) != NULL)
				PushDownLayer(prev_mp, true, layer_pt.level);

			SetLayer(layer_pt, hidden_layer);
			++layer_pt.row;
		}

		hidden_layer = (HiddenLayer*)hidden_layer->GetNext();
	}
}

bool NNMatrixModify::AvailableAddLayer(const MATRIX_POINT& insert_pt) const
{
	if (GetLayer(insert_pt)==NULL)
	{
		neuro_u32 last_level_size = GetLevelCount();
		if (last_level_size > 0) // 현재 하나 이상의 level이 있는 경우
		{
			if (insert_pt.level > last_level_size)
			{
				// 최소한 마지막 level 다음에서 클릭해야 한다.
				return false;
			}
			else if (insert_pt.level == last_level_size)	// 마지막 level 다음에서 클릭 했을 경우
			{
				// 최소한 row 위치는 이전 level 범위에 있어야 한다.
				if (insert_pt.row >= GetRowCount(last_level_size-1))
					return false;
			}
			else
			{
				// 정삼 범위 level에서 클릭 했을 경우 최소한 마지막 row 다음에서 클릭해야 한다.
				if (insert_pt.row > GetRowCount(insert_pt.level))
					return false;
			}
		}
	}
	return true;
}

AbstractLayer* NNMatrixModify::AddLayer(NeuralNetwork& network, network::_layer_type desire_hidden_type, MATRIX_POINT insert_pt, _POS_INFO_IN_LAYER::_pos_in_grid pos)
{
	if (!AvailableAddLayer(insert_pt))
		return NULL;

	bool insert_left_level = false;

	if (pos == _POS_INFO_IN_LAYER::_pos_in_grid::none)
	{
		if(GetLevelCount()==0)
			insert_pt.level = insert_pt.row = 0;	// 하나도 없는 상태에선 기본적으로 첫번째 level, row에 삽입 할수 있다.
	}
	else if (pos == _POS_INFO_IN_LAYER::_pos_in_grid::side_left)
	{
		if (insert_pt.level == 0)	// 입력 layer level 앞에 level을 추가할수 없다.
			return false;

		insert_left_level = true;
	}
	else if (pos == _POS_INFO_IN_LAYER::_pos_in_grid::side_right)
	{
		++insert_pt.level;
		insert_left_level = true;
	}
	else if (pos == _POS_INFO_IN_LAYER::_pos_in_grid::side_down)
	{
		++insert_pt.row;
	}

	network::_layer_type layer_type = insert_pt.level == 0 ? network::_layer_type::input : desire_hidden_type;

	AbstractLayer* insert_before = NULL;
	if(insert_pt.level < GetLevelCount())
	{
		neuro_u32 start_row = insert_pt.row;
		if (insert_left_level)
			start_row =0;	// level을 추가하려는 것이면, 지금 level의 첫번째 row부터 검색해야 함

		for (neuro_u32 level = insert_pt.level; level<insert_pt.level+2;level++)
		{
			for (neuro_u32 row = start_row, last_row=GetRowCount(level); row < last_row; row++)
			{
				AbstractLayer* layer = GetLayer(level, row);
				if(layer != NULL)
				{
					insert_before = layer;
					break;
				}
			}
			if (insert_before)
				break;
			start_row = 0;
		}
	}
	AbstractLayer* layer = network.AddLayer(layer_type, insert_before);
	if (layer == NULL)
		return NULL;

	bool updata_layer_layouts = false;

	if (!insert_left_level && GetLayer(insert_pt) != NULL)	
	{
		// 이미 layer가 있는 곳에 추가할 때 아래로 밀어 버린다.
		updata_layer_layouts = PushDownLayer(insert_pt, false);
	}
	else
	{
		// level을 추가하거나 비어 있는 곳이지만 현재 위치의 앞/뒤에 연결된 layer들이 있으면 뒤에거를 아래로 밀어 버린다.
		neuro_u32 right_level = insert_left_level ? insert_pt.level : insert_pt.level + 1;
		neuro_u32 last_level = GetLevelCount();
		for (; right_level < last_level; right_level++)
		{
			AbstractLayer* right = GetLayer(right_level, insert_pt.row);
			if (right != NULL)
			{
				AbstractLayer* right_input = GetFirstInputLayer(right);
				if (right_input)
				{
					MATRIX_POINT right_input_mp = GetLayerMatrixPoint(*right_input);
					if(right_input_mp.row== insert_pt.row)
						updata_layer_layouts = PushDownLayer(MATRIX_POINT(right_level, insert_pt.row), false);
				}
				break;
			}
		}
		if (insert_left_level)// 새로운 level을 추가할 경우
		{
			// 위에서 layer point 를 업데이트 해야면 일괄적으로 하면 되기때문에 InsertLevel에서 하지 않고
			// 그러지 않고 InsertLevel에서만 해도 되면 InsertLevel에서만 하고 updata_layer_layouts는 그대로 false로 두어
			// 더이상 MakeLayerLayouts 안하게 한다.
			InsertLevel(insert_pt.level, !updata_layer_layouts);
		}
	}

	SetLayer(insert_pt, layer);

	if(updata_layer_layouts)
		MakeLayerLayouts();

	return layer;
}

// 아래로 내리려는 layer를 기준으로 같이 내려야 하는 layer들을 2차원 scope 형태로 찾아내어 일괄적으로 내린다.
// 그리고, 움직일수 있는 것들만 움직이기 때문에 겹치는게 있는지 확인하고, 움직인 layer들의 위치들을 업데이트 해야한다.
bool NNMatrixModify::PushDownLayer(const MATRIX_POINT& pt, bool update_layer_pt, neuro_u32 level_bound, neuro_u32 down_row_count)
{
	if (down_row_count == 0)
		return false;

	_MOVING_SCOPE moving_scope;
	GetRelativeMovingVector(pt, level_bound, moving_scope);
	if (moving_scope.moving_level_scope_vector.size() == 0)
		return false;

	MoveRows(moving_scope, update_layer_pt, down_row_count);
	return true;
}

void NNMatrixModify::GetRelativeMovingVector(const MATRIX_POINT& pt, neuro_u32 level_bound, _MOVING_SCOPE& moving_scope)
{
	if (GetLayer(pt) == NULL)
		return;

	moving_scope.start_row = pt.row;

	_layer_set layer_set;

	if (level_bound == neuro_last32)
		level_bound = GetLevelCount();

	neuro_u32 start_level = pt.level;
	neuro_u32 last_level = pt.level + 1;

	neuro_u32 prev_start_level, prev_last_level;
	prev_start_level = prev_last_level = 0;

	for (neuro_u32 row = pt.row; ; row++)
	{
		neuro_u32 prev_add_layer = layer_set.size();

		// 이전의 무빙존의 level 범위의 layer 들은 모두 포함시킨다.
		bool checked_first_input = false;
		for (neuro_u32 level = start_level; level < last_level; level++)
		{
			AbstractLayer* layer = GetLayer(level, row);
			if (layer == NULL)
				continue;

			layer_set.insert(layer);

			//	첫번째 입력이 같은 row 위치에 있을 경우 범위를 확대 시킨다.
			if (checked_first_input)
				continue;

			checked_first_input = true;

			AbstractLayer* input = GetFirstInputLayer(layer);
			if (input)
			{
				MATRIX_POINT in_pt = GetLayerMatrixPoint(*input);
				if (in_pt.row == row)
				{
					start_level = in_pt.level;
					layer_set.insert(input);
				}
			}
		}

		// 이전 무빙존의 오른쪽에서 이전 무빙존의 layer를 첫번째 입력으로 가지는 layer를 찾아 오른쪽을 확대 시킨다.
		// 같이 움직여야 하기 때문
		for (neuro_u32 level = last_level; level < level_bound; level++)
		{
			AbstractLayer* layer = GetLayer(level, row);
			if (layer == NULL)
				continue;

			AbstractLayer* input = GetFirstInputLayer(layer);
			if (layer_set.find(input) == layer_set.end())
				continue;

			last_level = level + 1;
			layer_set.insert(layer);
		}

		// layer가 더이상 없다면 이제 그만 해도 된다.
		if (layer_set.size() == prev_add_layer)
			break;

		if (moving_scope.moving_level_scope_vector.size() == 0 || start_level < prev_start_level || last_level > prev_last_level)
		{
			moving_scope.moving_level_scope_vector.push_back({ start_level, last_level });

			prev_start_level = start_level;
			prev_last_level = last_level;
		}
	}
}

bool NNMatrixModify::DeleteLayers(NeuralNetwork& network, const MATRIX_SCOPE& scope, _std_u32_vector& deleted_uid_vector)
{
	if (!scope.IsValid())
		return false;

	_layer_vector erase_layer_vector;

	_std_u32_vector empty_level_vector;
	_std_u32_vector empty_row_vector;

	neuro_u32 last_level = min((neuro_u32)GetLevelCount(), scope.second.level);
	for (neuro_u32 level = scope.first.level; level < last_level; level++)
	{
		neuro_u32 last_row = min(GetRowCount(level), scope.second.row);
		for (neuro_u32 row = scope.first.row; row < last_row; row++)
		{
			AbstractLayer* layer = GetLayer(level, row);
			if (layer)
			{
				deleted_uid_vector.push_back(layer->uid);

				erase_layer_vector.push_back(layer);
				SetLayer(level, row, NULL);
			}
		}

		if (IsLevelEmpty(level))
			empty_level_vector.push_back(level);
	}

	for (neuro_u32 row = scope.first.row; row < scope.second.row; row++)
	{
		if (IsRowEmpty(row))
			empty_row_vector.push_back(row);
	}

	// 하나이상의 level 또는 row 가  다 지워지므로 다시 구성하는게 낮다.
	if (empty_level_vector.size() > 0 || empty_row_vector.size() > 0)
	{
		for (neuro_u32 i=0;i<erase_layer_vector.size();i++)
			network.DeleteLayer(erase_layer_vector[i]);

		BatchMatrixRemove(empty_level_vector, empty_row_vector);
		return true;
	}

	for (neuro_u32 i = 0; i<erase_layer_vector.size(); i++)
	{
		HiddenLayer* layer = (HiddenLayer*)erase_layer_vector[i];
		if (layer->GetLayerType() != network::_layer_type::input)
		{
			const _slice_input_vector& input_vector = layer->GetInputVector();
			for (neuro_u32 i = 0; i < input_vector.size(); i++)
				RemoveLinkInfo(input_vector[i].layer->uid, layer->uid);
		}

		const _hiddenlayer_set& output_set = layer->GetOutputSet();
		_hiddenlayer_set::const_iterator it = output_set.begin();
		for (; it != output_set.end(); it++)
			RemoveLinkInfo(layer->uid, (*it)->uid);

		network.DeleteLayer(layer);
	}

	return true;
}

void NNMatrixModify::DisconnectedInput(neuro_u32 layer_id, const _slice_input_vector& erased_input_vector)
{
	for (neuro_u32 i = 0; i < erased_input_vector.size(); i++)
		RemoveLinkInfo(erased_input_vector[i].layer->uid, layer_id);
}

// 요건 좀 복잡. 아니 많이 복잡... 나중에 하자!
bool NNMatrixModify::MoveLayerTo(NeuralNetwork& network, const MATRIX_POINT& layer_mp, const MATRIX_POINT& insert)
{
	return false;
/*	AbstractLayer* insert_layer = GetLayer(insert);
	if (!network.MoveLayerTo(layer, insert_layer))
		return false;

	// 영역을 움직이자!

	// 영역을 움직였으면, 각 layer의 입력 순서도 바꿔야 한다! 물론, concat 등 멀티 입력이 가능한 layer만!

	return true;*/
}

bool NNMatrixModify::ConnectTest(NeuralNetwork& network, const MATRIX_POINT& from_layer_mp, const MATRIX_POINT& to_layer_mp)
{
	// m_network_matrix 상에서 연결할 수 있는지 먼저 검사해야한다!
	if (!from_layer_mp.IsValid() || !to_layer_mp.IsValid())
		return false;

	AbstractLayer* from = GetLayer(from_layer_mp);
	AbstractLayer* to = GetLayer(to_layer_mp);

	if (from_layer_mp.level < to_layer_mp.level)
		return network.ConnectTest(from, to);
	else if (from_layer_mp.level == to_layer_mp.level)	// 같은 level에서의 연결은 lstm등 recurrent layer만 가능하다.
		return network.SideConnectTest(from, to);
	else
		return false;
}

bool NNMatrixModify::Connect(NeuralNetwork& network, const MATRIX_POINT& from_layer_mp, const MATRIX_POINT& to_layer_mp)
{
	if (!from_layer_mp.IsValid() || !to_layer_mp.IsValid())
		return false;

	AbstractLayer* from = GetLayer(from_layer_mp);
	AbstractLayer* to = GetLayer(to_layer_mp);

	// m_network_matrix 상에서 연결할 수 있는지 먼저 검사해야한다!
	if (from == NULL || to == NULL || to->GetLayerType()==network::_layer_type::input)
		return false;

	const AbstractLayer* remove_link_input=NULL;

	if (from_layer_mp.level < to_layer_mp.level)
	{
		AbstractLayer* insert_prev = NULL;

		if (((HiddenLayer*)to)->AvailableInputCount() > 1)
		{
			// concat와 같이 다중 입력이 있는 경우 입력 layer의 순서에 맞게 입력 벡터를 구성해야 한다.
			const network::_slice_input_vector& input_vector = ((HiddenLayer*)to)->GetInputVector();
			if (input_vector.size() > 0)
			{		
				MATRIX_POINT last_mp;
				for (neuro_u32 i = 0; i < input_vector.size(); i++)
				{
					MATRIX_POINT in_mp = GetLayerMatrixPoint(*input_vector[i].layer);
					if (in_mp.level < from_layer_mp.level || in_mp.level == from_layer_mp.level && in_mp.row < from_layer_mp.row)
						continue;

					if(insert_prev == NULL ||
						in_mp.level < last_mp.level || in_mp.level == last_mp.level && in_mp.row < last_mp.row)
					{
						insert_prev = input_vector[i].layer;
						last_mp = in_mp;
					}
				}
			}

		}
		else// 출력 layer가 1개의 입력만을 가지면 교체하는 것이기 때문에 미리 가지고 있는다
		{
			if (((HiddenLayer*)to)->GetMainInput() != NULL)
				remove_link_input = ((HiddenLayer*)to)->GetMainInput()->layer;
		}

		if (!network.Connect(from, to, insert_prev))
			return false;
	}
	else if (from_layer_mp.level == to_layer_mp.level)
	{
		if (((HiddenLayer*)to)->GetSideInput() != NULL)
			remove_link_input = ((HiddenLayer*)to)->GetSideInput();

		if (!network.SideConnect(from, to))
			return false;
	}
	else
		return false;

	if (remove_link_input)
		RemoveLinkInfo(remove_link_input->uid, to->uid);

	AddLinkInfo(from->uid, from_layer_mp, to->uid, to_layer_mp);

	return true;
}

bool NNMatrixModify::DisConnect(NeuralNetwork& network, const MATRIX_POINT& from_layer_mp, const MATRIX_POINT& to_layer_mp)
{
	AbstractLayer* from = GetLayer(from_layer_mp);
	AbstractLayer* to = GetLayer(to_layer_mp);

	if (from == NULL || to == NULL)
		return false;

	if (!network.DisConnect(from, to))
		return false;

	RemoveLinkInfo(from->uid, to->uid);
	return true;
}

AbstractLayer* NNMatrixModify::GetFirstInputLayer(const AbstractLayer* layer) const
{
	if (layer == NULL || layer->GetLayerType() == _layer_type::input)
		return NULL;

	const _slice_input_vector& in_vector = ((HiddenLayer*)layer)->GetInputVector();
	if (in_vector.size() == 0)
		return NULL;

	AbstractLayer* first_layer = in_vector[0].layer;
	MATRIX_POINT first_pos = GetLayerMatrixPoint(*first_layer);
	for (neuro_u32 i = 1; i < in_vector.size(); i++)
	{
		MATRIX_POINT in_pt = GetLayerMatrixPoint(*in_vector[i].layer);
		if (in_pt.level < first_pos.level || in_pt.row < first_pos.row)
		{
			first_layer = in_vector[i].layer;
			first_pos = in_pt;
		}
	}
	return first_layer;
}
