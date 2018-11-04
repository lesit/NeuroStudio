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

// �Է��� ���� layer�� ��� ��Ȯ�� ��ġ�� ���� ������ ������ ��ġ�� �������ش�.
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
		if (layer->GetMainInput() == NULL)	// �Է��� ���� ��� ���� �Է��� �����.
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
				if (next_layer && next_mp.level > cur_mp.level)	// ���� layer�� ���� level���� �����Ѵ�.
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

// loading, ���ΰ�ħ �������� �Ѵ�.
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

			if (vp.level_status == 2 || vp.level_status==3)	// ���� level���� ����
				++layer_pt.level;

			std::unordered_map<neuro_u32, AbstractLayer*>::const_iterator v_in_it = uid_layer_map.find(vp.virtual_input);
			if (v_in_it != uid_layer_map.end())
			{
				MATRIX_POINT in_mp = GetLayerMatrixPoint(*v_in_it->second);
				if (layer_pt.row < in_mp.row)
					layer_pt.row = in_mp.row + 1;
			}
			SetLayer(layer_pt, hidden_layer);
			if (vp.level_status == 1 || vp.level_status == 3)	// ���� layer�� ���� level���� �����Ѵ�.
			{
				++layer_pt.level;
				layer_pt.row = 0;
			}
			else
				++layer_pt.row;
		}
		else
		{
			// �Է� layer�� main connection�� ��� �� �Ǿ��� �� �� layer�� main connection �̴�.
			bool is_main_connection = main_connection_map.find(first_in) == main_connection_map.end();
			if(is_main_connection)
				main_connection_map.insert(first_in);

			const _slice_input_vector& in_vector = hidden_layer->GetInputVector();
			for (neuro_u32 i = 0; i < in_vector.size(); i++)
			{
				MATRIX_POINT in_pt = GetLayerMatrixPoint(*in_vector[i].layer);
				if (in_pt.level == layer_pt.level)	// �Է°� ���� level�̸� level �߰� �ʿ�
				{
					++layer_pt.level;
					layer_pt.row = 0;
					break;
				}
			}

			MATRIX_POINT first_in_pt = GetLayerMatrixPoint(*first_in);
			if (first_in_pt.row >= layer_pt.row) // ���� ù��° �Է� layer�� ��ġ���� ���� �ְų� ���ٸ�
			{
				// layer�� ù��° �Է� layer row��ġ�� ���ų� �� �Ʒ��� �ֵ��� ����
				layer_pt.row = first_in_pt.row;

				// ���� row ��ġ�ε� ���� 2 level �̻� ������ �ְ� �߰��� ��� ���� �ʴٸ� �� �Ʒ��� ������ �Ѵ�. 10 p
				if (first_in_pt.level + 1 < layer_pt.level)
				{
					for (neuro_u32 level = first_in_pt.level + 1; level < layer_pt.level; level++)
					{
						// ���� �߰��� �ٸ� layer�� �ִٸ�
						if (GetLayer(level, layer_pt.row) != NULL)	
						{
							++layer_pt.row;	// �Ѵܰ� �Ʒ��� ��ġ�ϵ��� ����
							break;
						}
					}
				}
			}
			else if (is_main_connection && first_in_pt.row < layer_pt.row)
			{	// ���� �����ε� ù��° �Է� layer���� �Ʒ��� �ִٸ� �Է��� ����������
				PushDownLayer(first_in_pt, true, layer_pt.level, layer_pt.row - first_in_pt.row);
			}
			/*	item_size = { 100,50 }, margin_in_grid = { 50, 25 } ���� ������ �ٷ� �տ� layer�� �־ �̻����� ����
			������ �� layer ���� �ٸ� ����� ���� ��찡 ���� ������ �������� �����ֱ� ���� �׳� �̴�� ����Ѵ�.
			*/
			// ���� row��ġ���� �տ� �� �ٸ� layer�� �ִٸ� �� layer�� �Ʒ��� �о������
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
		if (last_level_size > 0) // ���� �ϳ� �̻��� level�� �ִ� ���
		{
			if (insert_pt.level > last_level_size)
			{
				// �ּ��� ������ level �������� Ŭ���ؾ� �Ѵ�.
				return false;
			}
			else if (insert_pt.level == last_level_size)	// ������ level �������� Ŭ�� ���� ���
			{
				// �ּ��� row ��ġ�� ���� level ������ �־�� �Ѵ�.
				if (insert_pt.row >= GetRowCount(last_level_size-1))
					return false;
			}
			else
			{
				// ���� ���� level���� Ŭ�� ���� ��� �ּ��� ������ row �������� Ŭ���ؾ� �Ѵ�.
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
			insert_pt.level = insert_pt.row = 0;	// �ϳ��� ���� ���¿��� �⺻������ ù��° level, row�� ���� �Ҽ� �ִ�.
	}
	else if (pos == _POS_INFO_IN_LAYER::_pos_in_grid::side_left)
	{
		if (insert_pt.level == 0)	// �Է� layer level �տ� level�� �߰��Ҽ� ����.
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
			start_row =0;	// level�� �߰��Ϸ��� ���̸�, ���� level�� ù��° row���� �˻��ؾ� ��

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
		// �̹� layer�� �ִ� ���� �߰��� �� �Ʒ��� �о� ������.
		updata_layer_layouts = PushDownLayer(insert_pt, false);
	}
	else
	{
		// level�� �߰��ϰų� ��� �ִ� �������� ���� ��ġ�� ��/�ڿ� ����� layer���� ������ �ڿ��Ÿ� �Ʒ��� �о� ������.
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
		if (insert_left_level)// ���ο� level�� �߰��� ���
		{
			// ������ layer point �� ������Ʈ �ؾ߸� �ϰ������� �ϸ� �Ǳ⶧���� InsertLevel���� ���� �ʰ�
			// �׷��� �ʰ� InsertLevel������ �ص� �Ǹ� InsertLevel������ �ϰ� updata_layer_layouts�� �״�� false�� �ξ�
			// ���̻� MakeLayerLayouts ���ϰ� �Ѵ�.
			InsertLevel(insert_pt.level, !updata_layer_layouts);
		}
	}

	SetLayer(insert_pt, layer);

	if(updata_layer_layouts)
		MakeLayerLayouts();

	return layer;
}

// �Ʒ��� �������� layer�� �������� ���� ������ �ϴ� layer���� 2���� scope ���·� ã�Ƴ��� �ϰ������� ������.
// �׸���, �����ϼ� �ִ� �͵鸸 �����̱� ������ ��ġ�°� �ִ��� Ȯ���ϰ�, ������ layer���� ��ġ���� ������Ʈ �ؾ��Ѵ�.
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

		// ������ �������� level ������ layer ���� ��� ���Խ�Ų��.
		bool checked_first_input = false;
		for (neuro_u32 level = start_level; level < last_level; level++)
		{
			AbstractLayer* layer = GetLayer(level, row);
			if (layer == NULL)
				continue;

			layer_set.insert(layer);

			//	ù��° �Է��� ���� row ��ġ�� ���� ��� ������ Ȯ�� ��Ų��.
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

		// ���� �������� �����ʿ��� ���� �������� layer�� ù��° �Է����� ������ layer�� ã�� �������� Ȯ�� ��Ų��.
		// ���� �������� �ϱ� ����
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

		// layer�� ���̻� ���ٸ� ���� �׸� �ص� �ȴ�.
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

	// �ϳ��̻��� level �Ǵ� row ��  �� �������Ƿ� �ٽ� �����ϴ°� ����.
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

// ��� �� ����. �ƴ� ���� ����... ���߿� ����!
bool NNMatrixModify::MoveLayerTo(NeuralNetwork& network, const MATRIX_POINT& layer_mp, const MATRIX_POINT& insert)
{
	return false;
/*	AbstractLayer* insert_layer = GetLayer(insert);
	if (!network.MoveLayerTo(layer, insert_layer))
		return false;

	// ������ ��������!

	// ������ ����������, �� layer�� �Է� ������ �ٲ�� �Ѵ�! ����, concat �� ��Ƽ �Է��� ������ layer��!

	return true;*/
}

bool NNMatrixModify::ConnectTest(NeuralNetwork& network, const MATRIX_POINT& from_layer_mp, const MATRIX_POINT& to_layer_mp)
{
	// m_network_matrix �󿡼� ������ �� �ִ��� ���� �˻��ؾ��Ѵ�!
	if (!from_layer_mp.IsValid() || !to_layer_mp.IsValid())
		return false;

	AbstractLayer* from = GetLayer(from_layer_mp);
	AbstractLayer* to = GetLayer(to_layer_mp);

	if (from_layer_mp.level < to_layer_mp.level)
		return network.ConnectTest(from, to);
	else if (from_layer_mp.level == to_layer_mp.level)	// ���� level������ ������ lstm�� recurrent layer�� �����ϴ�.
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

	// m_network_matrix �󿡼� ������ �� �ִ��� ���� �˻��ؾ��Ѵ�!
	if (from == NULL || to == NULL || to->GetLayerType()==network::_layer_type::input)
		return false;

	const AbstractLayer* remove_link_input=NULL;

	if (from_layer_mp.level < to_layer_mp.level)
	{
		AbstractLayer* insert_prev = NULL;

		if (((HiddenLayer*)to)->AvailableInputCount() > 1)
		{
			// concat�� ���� ���� �Է��� �ִ� ��� �Է� layer�� ������ �°� �Է� ���͸� �����ؾ� �Ѵ�.
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
		else// ��� layer�� 1���� �Է¸��� ������ ��ü�ϴ� ���̱� ������ �̸� ������ �ִ´�
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
