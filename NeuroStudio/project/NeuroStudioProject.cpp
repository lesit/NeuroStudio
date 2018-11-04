#include "stdafx.h"

#include "NeuroStudioProject.h"

#include "storage/FileDeviceAdaptor.h"

#include "BindingViewManager.h"

#include "LoadNSProject.h"
#include "SaveNSProject.h"

#include "util/FileUtil.h"

using namespace np;
using namespace np::project;

NeuroStudioProject::NeuroStudioProject(ProjectManager& manager, DeepLearningDesignViewManager& design_view)
	: m_manager(manager)
	, m_design_view(design_view)
	, m_sim_manager(m_ns_manager)
{
	m_nd_factory=NULL;

	m_design_view.SetProject(this);

	LoadViews();
}

NeuroStudioProject::~NeuroStudioProject()
{
	m_design_view.SetProject(NULL);

	delete m_nd_factory;
	m_nd_factory=NULL;
}

#ifdef _DEBUG
void NeuroStudioProject::SampleProject()
{
	m_ns_manager.SampleCreate();
	LoadViews();
}
#endif

void NeuroStudioProject::LoadViews()
{
	m_design_view.LoadView();
}

void NeuroStudioProject::SaveViews()
{
	m_design_view.SaveView();
}

device::IODeviceFactory* NeuroStudioProject::CreateFileDeviceFactory(const char* strFilePath)
{
	return new device::FileDeviceFactory(strFilePath);
}

const char* NeuroStudioProject::GetNetworkFilePath() const
{
	if(m_nd_factory==NULL || m_nd_factory->GetType()!=device::_device_type::file)
		return NULL;

	device::FileDeviceFactory* file_device=(device::FileDeviceFactory*)m_nd_factory;
	return file_device->GetFilePath();
}

void NeuroStudioProject::NewProject()
{
	NewNeuroSystem();
}

bool NeuroStudioProject::SaveProject()
{
	if (!m_project_file_path.empty())
		return SaveProject(m_project_file_path.c_str());
	return true;
}

bool NeuroStudioProject::SaveProject(const char* project_path)
{
	// ������Ʈ�� �����Ҷ� �ݵ�� ��Ʈ��ũ�� ��������.
	bool bSaveAs=m_project_file_path.compare(project_path) != 0;
	m_project_file_path = project_path;
	if (!SaveNetworkStructure(bSaveAs))
		return false;

	np::project::SaveNSProject project(*this);
	return project.Save(project_path);
}

bool NeuroStudioProject::LoadProject(const char* project_path)
{
	if(strlen(project_path)==0)
		return false;

	// ���ο�� �ε��ϱ� ���� ���� ������Ʈ�� �����ؾ� ���ٵ�...
	// �׷�����, �����Ȱ� �ֳ� ���� Ȯ���ؾ��Ѵ�.
	if(!m_project_file_path.empty())	
	{
		if(!SaveProject(m_project_file_path.c_str()))
			return false;
	}

	np::project::LoadNSProject load(*this);
	bool ret = load.Load(project_path);
	if(ret)
		m_project_file_path = project_path;

	LoadViews();
	return ret;
}

void NeuroStudioProject::NewNeuroSystem()
{
	DEBUG_OUTPUT(L"");

	m_ns_manager.NewSystem();
	if (m_nd_factory)
	{
		delete m_nd_factory;
		m_nd_factory=NULL;
	}

	LoadViews();
}

void NeuroStudioProject::NewNetworkStructure()
{
	DEBUG_OUTPUT(L"");

	m_ns_manager.NetworkNew();
	if (m_nd_factory)
	{
		delete m_nd_factory;
		m_nd_factory = NULL;
	}
	LoadViews();
}

bool NeuroStudioProject::OpenNetworkStructure(const char* new_nn_path, bool load_view)
{
	if (new_nn_path == NULL || strlen(new_nn_path) == 0)
		return false;

	device::IODeviceFactory* nd_desc = CreateFileDeviceFactory(new_nn_path);
	if (!OpenNetworkStructure(nd_desc, load_view))
	{
		if(nd_desc)
			delete nd_desc;
		return false;
	}

	return true;
}

bool NeuroStudioProject::OpenNetworkStructure(device::IODeviceFactory* nd_desc, bool load_view)
{
	if (!nd_desc)
		return false;

	bool bRet=false;
	if (nd_desc)
	{
		bRet = m_ns_manager.NetworkLoad(nd_desc);
		if (bRet)
		{
			DEBUG_OUTPUT(L"successed to load network");
			if (m_nd_factory != nd_desc)
				delete m_nd_factory;

			m_nd_factory = nd_desc;
		}
	}
	else
		DEBUG_OUTPUT(L"failed to load network");

	if(load_view)
		LoadViews();
	return bRet;
}

bool NeuroStudioProject::SaveNetworkStructure(bool bSaveAs, bool bReload, bool* bCancel)
{
	DEBUG_OUTPUT(bSaveAs ? L"SaveAs":L"Save");

	SaveViews();

	if (!m_nd_factory)
		bSaveAs=true;

	bool bRet=true;
	if(!bSaveAs)	
	{	// ������Ʈ ����, ��Ʈ��ũ ����, �ùķ��̼��� ����(������ ������ ��쿡��)

		// ������ ���(�Ǵ� device)�� �����Ҷ���, ��Ʈ��ũ�� ���µǾ������� �����Ұ� ������ �����ؾ��Ѵ�.
		// �ֳĸ�, ����ִ� �� ��ü�� ������ �����̱� ����
		if(m_ns_manager.GetNetwork())	// ���µȰ� �������� �����Ѵ�.
			bRet = m_ns_manager.NetworkSave(bReload);
	}
	// save as
	else
	{	// ������Ʈ ����, ��Ʈ��ũ ����, �ùķ��̼��� ����(������ ������ ��쿡��)
		std::string default_name = util::FileUtil::GetNameFromFileName<char>(m_project_file_path.c_str());
		std::string new_nn_path = m_manager.RequestNetworkSavePath(default_name.c_str());
		if(new_nn_path.empty())
		{
			DEBUG_OUTPUT(L"no network save path");
			if(bCancel)
				*bCancel=true;

			return false;
		}

		device::IODeviceFactory* new_device_factory= CreateFileDeviceFactory(new_nn_path.c_str());
		if(!new_device_factory)
		{
			DEBUG_OUTPUT(L"failed to create file device");
			return false;
		}

		bRet = m_ns_manager.NetworkSaveAs(*new_device_factory, 4*1024, bReload);
		if (bRet)
		{
			delete m_nd_factory;
			m_nd_factory = new_device_factory;
		}
		else
			delete new_device_factory;
	}

	if(!bRet)
		DEBUG_OUTPUT(L"failed to save network");

	LoadViews();
	DEBUG_OUTPUT(L"successed to save network");
	return bRet;
}
