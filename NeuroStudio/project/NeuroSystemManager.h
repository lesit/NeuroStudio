#pragma once

#include "NeuroKernel/network/NeuralNetwork.h"
#include "NeuroData/model/DataProviderModel.h"

namespace np
{
	namespace project
	{
		namespace network_ready_error
		{
			enum class _error_type { no_network, layer_error };
			class ReadyError
			{
			public:
				virtual ~ReadyError() {}

				virtual _error_type GetType() const = 0;
				virtual const wchar_t* GetString() const = 0;
			};

			class NetworkError : public ReadyError
			{
			public:
				_error_type GetType() const override { return _error_type::no_network; }

				const wchar_t* GetString() const override { return L"no network"; }
			};

			struct _LAYER_ERROR_INFO
			{
				_LAYER_ERROR_INFO(const network::AbstractLayer* layer, const wchar_t* msg)
				{
					this->layer = layer;
					this->msg = msg;
				}
				const network::AbstractLayer* layer;
				const wchar_t* msg;
			};
			typedef std::vector<_LAYER_ERROR_INFO> _layer_error_vector;

			class LayersError : public ReadyError
			{
			public:
				_error_type GetType() const override { return _error_type::layer_error; }
				const wchar_t* GetString() const override { return L"layer error"; }

				_layer_error_vector layer_error_vector;
			};
		}

		class NeuroSystemManager
		{
		public:
			NeuroSystemManager();
			virtual ~NeuroSystemManager();

			bool SampleCreate();

			void NewSystem();

			bool NetworkNew();
			bool NetworkLoad(device::IODeviceFactory* nd_desc);
			bool NetworkSave(bool bReload);
			bool NetworkSaveAs(device::IODeviceFactory& nd_desc, neuro_u32 block_size = 4 * 1024, bool bReload=true);

			network_ready_error::ReadyError* ReadyValidationCheck() const;

			dp::model::ProviderModelManager& GetProvider() { return m_provider; }
			const dp::model::ProviderModelManager& GetProvider() const { return m_provider; }

			network::NeuralNetwork* GetNetwork(){ return m_network; }
			const network::NeuralNetwork* GetNetwork() const{ return m_network; }

		protected:
			void CloseAll();

		private:
			dp::model::ProviderModelManager m_provider;

			network::NeuralNetwork* m_network;
		};
	}
}
