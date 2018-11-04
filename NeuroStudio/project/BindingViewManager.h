#pragma once

#include "common.h"
#include "gui/shape.h"
#include "gui/line.h"

#include "NeuroBindingLink.h"

#include "NeuroStudioProject.h"
#include "NeuroData/model/AbstractReaderModel.h"
#include "NeuroData/model/AbstractProducerModel.h"

#include "NeuroKernel/network/NeuralNetwork.h"

#include "LastSetLayerEntryVector.h"
#include "../network/NNMatrixModify.h"

namespace np
{
	using namespace np::dp::model;
	using namespace np::network;
	using namespace gui;

	namespace project
	{
		class AbstractBindingViewManager;
		class AbstractBindedViewManager
		{
		public:
			AbstractBindedViewManager(AbstractBindingViewManager& binding_view);
			virtual ~AbstractBindedViewManager() {}

			AbstractBindingViewManager& GetBindingView() { return m_binding_view; }

			virtual void Activated() {}
			virtual void Deactivated() {}

			project::NeuroStudioProject* GetProject();
			const project::NeuroStudioProject* GetProject() const;

			project::NeuroSystemManager* GetNSManager();
			const project::NeuroSystemManager* GetNSManager() const;

			virtual void LoadView() {}
			virtual void SaveView() {}

			virtual void ResetSelect() {}
			virtual void RefreshView() = 0;

		protected:
			AbstractBindingViewManager& m_binding_view;
		};

		struct _BINDING_SOURCE_MODEL
		{
			NP_POINT from_point;
			NeuroBindingModel* from;
			NeuroBindingModel* to;
		};
		typedef std::vector<_BINDING_SOURCE_MODEL> _binding_source_vector;

		class DataViewManager : public AbstractBindedViewManager
		{
		public:
			DataViewManager(AbstractBindingViewManager& binding_view)
				: AbstractBindedViewManager(binding_view)
			{}
			virtual ~DataViewManager() {}

			virtual void GetBindedModelVector(_binding_source_vector& model_vector) const = 0;
		};

		class NetworkViewManager : public AbstractBindedViewManager
		{
		public:
			NetworkViewManager(AbstractBindingViewManager& binding_view)
				: AbstractBindedViewManager(binding_view)
			{}
			virtual ~NetworkViewManager() {}

			virtual bool GetDataBoundLinePoints(const NP_POINT& from_point, const NeuroBindingModel& model, bool& is_hide, gui::_bezier_pt_vector& points) const = 0;
			virtual MATRIX_POINT GetLayerLocation(const AbstractLayer& layer) const = 0;

			virtual void SelectNetworkLayer(network::AbstractLayer* layer) = 0;
		};

		class AbstractBindingViewManager
		{
		public:
			AbstractBindingViewManager(const std::vector<DataViewManager*>& source_vector, NetworkViewManager& network_view);
			virtual ~AbstractBindingViewManager() {}

			void LoadView();
			void SaveView();

			void ClearBindingLineVector();
			void MakeBindingLineVector();
			virtual void RefreshBindingViews();

			void RefreshNetworkView();

			const _binding_link_vector& GetBindingLinkVector() const { return m_binding_link_vector; }

			virtual _line_arrow_type GetLineArrowType() const { return _line_arrow_type::end; }

			void SetBindingMouseoverLink(const _NEURO_BINDING_LINK* link);
			const _NEURO_BINDING_LINK* GetBindingMouseoverLink() const { return m_mouse_over_link; }

			void SetBindingSelectLink(const _NEURO_BINDING_LINK* link);
			const _NEURO_BINDING_LINK* GetBindingSelectedLink() const { return m_selected_link; }

			void InitSelection(AbstractBindedViewManager* exclude = NULL);
			void SelectNetworkLayer(network::AbstractLayer* layer) { m_network_view.SelectNetworkLayer(layer); }

			void SetDragStartPoint(const NP_POINT& pt) 
			{
				m_is_dragged = true;
				m_drag_start_point = pt; 
			}
			void SetDragEnd()
			{
				m_is_dragged = false;
				memset(&m_drag_start_point, 0, sizeof(NP_POINT));
			}

			bool GetDragStartPoint(NP_POINT& pt) const 
			{
				if (!m_is_dragged)
					return false;
				pt = m_drag_start_point; 
				return true;
			}

			virtual NeuroStudioProject* GetProject() = 0;
			inline const NeuroStudioProject* GetProject() const
			{
				return const_cast<AbstractBindingViewManager*>(this)->GetProject();
			}

		protected:
			NetworkViewManager& m_network_view;

		private:
			std::vector<DataViewManager*> m_source_vector;

			bool m_is_dragged;
			NP_POINT m_drag_start_point;

			_binding_link_vector m_binding_link_vector;

			const _NEURO_BINDING_LINK* m_mouse_over_link;
			const _NEURO_BINDING_LINK* m_selected_link;
		};

		class DeepLearningDesignViewManager : public AbstractBindingViewManager
		{
		public:
			DeepLearningDesignViewManager(const std::vector<DataViewManager*>& source_vector, NetworkViewManager& network_view)
				: AbstractBindingViewManager(source_vector, network_view)
			{}
			virtual ~DeepLearningDesignViewManager() {}

			NeuroStudioProject* GetProject() override { return m_project; }

		protected:
			friend class NeuroStudioProject;
			virtual void SetProject(NeuroStudioProject* project)
			{
				m_project = project;
			}

			NeuroStudioProject* m_project;
		};
	}
}
