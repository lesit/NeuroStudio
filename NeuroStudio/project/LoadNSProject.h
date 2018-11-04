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
		class LoadNSProject
		{
		public:
			LoadNSProject(NeuroStudioProject& project);

			bool Load(const char* strFilePath);

		protected:
			void LoadNeuralNetwork(const XERCES_CPP_NAMESPACE::DOMElement* elem);
			void LoadDataPreprocessor(const XERCES_CPP_NAMESPACE::DOMElement* elem);
			void LoadProvider(const XERCES_CPP_NAMESPACE::DOMElement* elem, dp::model::DataProviderModel& provider);
			void LoadDataReader(const XERCES_CPP_NAMESPACE::DOMElement* elem, dp::model::DataProviderModel& provider);
			void LoadProducer(const XERCES_CPP_NAMESPACE::DOMElement* elem, dp::model::DataProviderModel& provider);

			void LoadSimulation(const XERCES_CPP_NAMESPACE::DOMElement* elem);
			void LoadSimResultViewList(const XERCES_CPP_NAMESPACE::DOMElement* elem);
			void LoadSimEnvElem(const XERCES_CPP_NAMESPACE::DOMElement* elem);

			void LoadSimTrainData(const XERCES_CPP_NAMESPACE::DOMElement* elem);
			void LoadSimPredictData(const XERCES_CPP_NAMESPACE::DOMElement* elem);

			void LoadSimData(const XERCES_CPP_NAMESPACE::DOMElement* elem, dp::preprocessor::_uid_datanames_map& provider_data);
		private:
			NeuroStudioProject& m_project;
			NeuroSystemManager& m_nsManager;
			dp::model::ProviderModelManager& m_provider;

			std::string m_proj_dir;
		};
	}
}
