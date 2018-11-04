#pragma once

#include "NeuroKernel/network/NeuralNetwork.h"
#include "gui/shape.h"

#include "NetworkMatrix.h"
namespace np
{
	using namespace network;

	namespace network
	{
		class NNMatrixModify : public NetworkMatrix
		{
		public:
			NNMatrixModify();
			virtual ~NNMatrixModify();

			void NetworkChanged(network::NeuralNetwork* network);
			void UpdateNetwork(network::NeuralNetwork& network);

			bool AvailableAddLayer(const MATRIX_POINT& insert_pt) const;
			AbstractLayer* AddLayer(network::NeuralNetwork& network, network::_layer_type desire_hidden_type, MATRIX_POINT insert_pt, _POS_INFO_IN_LAYER::_pos_in_grid pos);
			bool DeleteLayer(network::NeuralNetwork& network, AbstractLayer* layer, _std_u32_vector& deleted_uid_vector)
			{
				if (layer == NULL)
					return false;

				MATRIX_POINT layer_mp = GetLayerMatrixPoint(*layer);
				return DeleteLayers(network, { layer_mp , MATRIX_POINT(layer_mp.level + 1, layer_mp.row + 1) }, deleted_uid_vector);
			}
			bool DeleteLayers(network::NeuralNetwork& network, const MATRIX_SCOPE& scope, _std_u32_vector& deleted_uid_vector);
			bool MoveLayerTo(network::NeuralNetwork& network, const MATRIX_POINT& layer_mpr, const MATRIX_POINT& insert);

			bool ConnectTest(network::NeuralNetwork& network, const MATRIX_POINT& from_layer_mp, const MATRIX_POINT& to_layer_mp);
			bool Connect(network::NeuralNetwork& network, const MATRIX_POINT& from_layer_mp, const MATRIX_POINT& to_layer_mp);
			bool DisConnect(network::NeuralNetwork& network, const MATRIX_POINT& from_layer_mp, const MATRIX_POINT& to_layer_mp);

			void DisconnectedInput(neuro_u32 layer_id, const _slice_input_vector& erased_input_vector);

		protected:
			neuro_u32 GetItemUid(const AbstractLayer& item) const override;
			void GetInputItemVector(const AbstractLayer& item, std::vector<const AbstractLayer*>& item_vector) const override;
			void GetOutputItemVector(const AbstractLayer& item, std::vector<const AbstractLayer*>& item_vector) const override;

			void MakeMatrix(const _LINKED_LAYER& input, const _LINKED_LAYER& hidden);
			bool PushDownLayer(const MATRIX_POINT& pt, bool update_layer_pt, neuro_u32 level_bound = neuro_last32, neuro_u32 down_row_count = 1);
			void GetRelativeMovingVector(const MATRIX_POINT& pt, neuro_u32 last_level, NetworkMatrix::_MOVING_SCOPE& moving_scope);

			AbstractLayer* GetFirstInputLayer(const AbstractLayer* layer) const;
		};
	}
}
