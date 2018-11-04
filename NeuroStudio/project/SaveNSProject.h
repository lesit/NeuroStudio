#pragma once

#include "NeuroStudioProject.h"
#include <xercesc/dom/DOMElement.hpp>

#include "NeuroData/model/BinaryReaderModel.h"
#include "NeuroData/model/TextReaderModel.h"
using namespace np::dp::model;

namespace np
{
	namespace project
	{
		class SaveNSProject
		{
		public:
			SaveNSProject(NeuroStudioProject& project);

			bool Save(const char* strFilePath);

		protected:
			XERCES_CPP_NAMESPACE::DOMElement* CreateNeuralNetworkElem();
			XERCES_CPP_NAMESPACE::DOMElement* CreateDataPreprocessorElem();
			XERCES_CPP_NAMESPACE::DOMElement* CreateProviderElem(const wchar_t* provider_name, const dp::model::DataProviderModel& provider);
			XERCES_CPP_NAMESPACE::DOMElement* CreateReaderElem(const dp::model::AbstractReaderModel& model);

			XERCES_CPP_NAMESPACE::DOMElement* CreateProducerElem(const dp::model::AbstractProducerModel& model);

			XERCES_CPP_NAMESPACE::DOMElement* CreateSimulationElem();
			XERCES_CPP_NAMESPACE::DOMElement* CreateSimResultViewListElem();
			XERCES_CPP_NAMESPACE::DOMElement* CreateSimEnvElem();

			XERCES_CPP_NAMESPACE::DOMElement* CreateSimTrainDataElem();
			XERCES_CPP_NAMESPACE::DOMElement* CreateSimPredictDataElem();

			void CompositeSimDataElem(XERCES_CPP_NAMESPACE::DOMElement* elem, const dp::preprocessor::_uid_datanames_map& provider_data);
		private:
			NeuroStudioProject& m_project;
			const dp::model::ProviderModelManager& m_provider;

			std::string m_proj_dir;

			XERCES_CPP_NAMESPACE::DOMDocument* m_document;
		};
	}
}
