#pragma once

#include "NeuroSystemManager.h"
#include "SimDefinition.h"

namespace np
{
	namespace project
	{
		class AbstractBindedViewManager;
		typedef std::vector<AbstractBindedViewManager*> _view_vector;

		class ProjectManager
		{
		public:
			virtual bool RequestSaveProject() = 0;
			virtual std::string RequestNetworkSavePath(const char* name = "") const = 0;

//			virtual std::wstring GetTempNetworkDevicePath() const = 0;
		};

		class DeepLearningDesignViewManager;
		class NeuroStudioProject
		{
		public:
			NeuroStudioProject(ProjectManager& manager, DeepLearningDesignViewManager& design_view);
			virtual ~NeuroStudioProject();

			const char* GetProjectFilePath() const {return m_project_file_path.c_str();}

			void NewProject();
			bool SaveProject();
			bool SaveProject(const char* project_path);
			bool LoadProject(const char* project_path);

			void NewNeuroSystem();

			void NewNetworkStructure();
			bool OpenNetworkStructure(const char* new_nn_path, bool load_view=true);
			bool OpenNetworkStructure(device::IODeviceFactory* nd_desc, bool load_view = true);
			bool SaveNetworkStructure(bool bSaveAs=false, bool bReload=true, bool* bCancel=NULL);

			device::IODeviceFactory* CreateFileDeviceFactory(const char* strFilePath);

			bool HasDeviceFactory(){return m_nd_factory!=NULL;}
			device::IODeviceFactory* GetNNDeviceFactory(){ return m_nd_factory; }

			const char* GetNetworkFilePath() const;

			NeuroSystemManager& GetNSManager(){return m_ns_manager;}
			SimDefinition& GetSimManager(){return m_sim_manager;}

			network_ready_error::ReadyError* ReadyValidationCheck() const {
				return m_ns_manager.ReadyValidationCheck();
			}

#ifdef _DEBUG
			void SampleProject();
#endif
		protected:
			void LoadViews();
			void SaveViews();

		protected:
			DeepLearningDesignViewManager& m_design_view;

			NeuroSystemManager m_ns_manager;
			SimDefinition m_sim_manager;

			std::string m_project_file_path;

			ProjectManager& m_manager;

		private:
			device::IODeviceFactory* m_nd_factory;
		};
	}
}
