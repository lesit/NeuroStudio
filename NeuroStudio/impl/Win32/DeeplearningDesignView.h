#pragma once

#include "DesignPreprocessorWnd.h"
#include "DesignNetworkWnd.h"
#include "ModelPropertyWnd.h"
#include "DesignErrorOutputPane.h"

#include "TitleWnd.h"

// DeeplearningDesignView ���Դϴ�.

class DeeplearningDesignView : public CView, public DeepLearningDesignViewManager
{
public:
	virtual ~DeeplearningDesignView();

	DesignPreprocessorWnd& GetProviderWnd() { return m_preprocessorWnd; }
	DesignNetworkWnd& GetNetworkWnd(){ return m_networkWnd; }

	DECLARE_DYNCREATE(DeeplearningDesignView)

	ModelPropertyWnd& GetPropertyPane();
	DesignErrorOutputPane& GetErrorOutputWnd();

protected:
	DeeplearningDesignView();           // ���� ����⿡ ���Ǵ� protected �������Դϴ�.

	void SetProject(NeuroStudioProject* project) override
	{
		__super::SetProject(project);

		if (m_project)
			GetPropertyPane().SetModelProperty(m_network_view, m_project->GetNSManager().GetNetwork());
		else
			GetPropertyPane().Clear();

		GetErrorOutputWnd().Clear();
	}

public:
	virtual void OnDraw(CDC* pDC);      // �� �並 �׸��� ���� �����ǵǾ����ϴ�.
#ifdef _DEBUG
	virtual void AssertValid() const;
#ifndef _WIN32_WCE
	virtual void Dump(CDumpContext& dc) const;
#endif
#endif

protected:
	DECLARE_MESSAGE_MAP()
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnSize(UINT nType, int cx, int cy);

private:
	CTitleWnd m_preprocessorTitle;
	DesignPreprocessorWnd m_preprocessorWnd;

	CTitleWnd m_networkTitle;
	DesignNetworkWnd m_networkWnd;
};
