// AnalysisWnd.cpp : implementation file
//

#include "stdafx.h"
#include "AnalysisWnd.h"

#include "gui/win32/WinUtil.h"

// CAnalysisWnd

IMPLEMENT_DYNAMIC(CAnalysisWnd, CWnd)

CAnalysisWnd::CAnalysisWnd()
{
	m_has_argmax_accuracy = true;
	m_has_second_history = false;

	m_backBrush.CreateSolidBrush(RGB(255, 255, 255));
}

CAnalysisWnd::~CAnalysisWnd()
{
}

#define IDC_LIST_EPOCH_HISTORY		WM_USER+1
#define IDC_GRAPH_EPOCH_LOSS		WM_USER+2
#define IDC_GRAPH_TEST_EPOCH_LOSS   WM_USER+3

BEGIN_MESSAGE_MAP(CAnalysisWnd, CWnd)
	ON_WM_CREATE()
	ON_WM_SIZE()
	ON_WM_CTLCOLOR()
END_MESSAGE_MAP()

int CAnalysisWnd::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CWnd::OnCreate(lpCreateStruct) == -1)
		return -1;

	CRect rcDummy(0, 0, 0, 0);
	m_ctrEpochHistoryStatic.Create(L" epoch history", WS_CHILD | WS_VISIBLE | SS_CENTERIMAGE, rcDummy, this);
	m_ctrEpochHistoryList.Create(WS_CHILD | WS_VISIBLE | WS_BORDER | LVS_REPORT | LVS_SINGLESEL | LVS_SHOWSELALWAYS, rcDummy, this, IDC_LIST_EPOCH_HISTORY);
	m_ctrEpochHistoryList.SetExtendedStyle(m_ctrEpochHistoryList.GetExtendedStyle() | LVS_EX_GRIDLINES | LVS_EX_FULLROWSELECT);

	m_ctrEpochGraphStatic.Create(L" loss graph", WS_CHILD | WS_VISIBLE | SS_CENTERIMAGE, rcDummy, this);
	m_ctrEpochGraphWnd.Create(NULL, NULL, WS_CHILD | WS_VISIBLE | WS_BORDER, rcDummy, this, IDC_GRAPH_EPOCH_LOSS);

	m_ctrTestEpochGraphStatic.Create(L" test graph", WS_CHILD | WS_VISIBLE | SS_CENTERIMAGE, rcDummy, this);
	m_ctrTestEpochGraphWnd.Create(NULL, NULL, WS_CHILD | WS_VISIBLE | WS_BORDER, rcDummy, this, IDC_GRAPH_TEST_EPOCH_LOSS);

	return 0;
}

HBRUSH CAnalysisWnd::OnCtlColor(CDC* pDC, CWnd* pWnd, UINT nCtlColor)
{
	HBRUSH hbr = __super::OnCtlColor(pDC, pWnd, nCtlColor);

	if (nCtlColor == CTLCOLOR_STATIC)
	{
		pDC->SetBkColor(RGB(255, 255, 255));
		return m_backBrush;
	}

	return hbr;
}

void CAnalysisWnd::OnSize(UINT nType, int cx, int cy)
{
	CWnd::OnSize(nType, cx, cy);

	CRect rcClientRect;
	GetClientRect(&rcClientRect);
//	rcClientRect.DeflateRect({ 5,5 });

	CRect rcLabel(rcClientRect.left, rcClientRect.top, 300, 20);

	m_ctrEpochHistoryStatic.MoveWindow(rcLabel);

	CRect rcList = rcLabel;
	rcList.top = rcLabel.bottom;
	rcList.bottom = rcClientRect.bottom;
	m_ctrEpochHistoryList.MoveWindow(rcList);

	CRect rcGraph = rcList;
	rcGraph.left = rcList.right + 10;
	if (m_has_second_history)
		rcGraph.right = rcGraph.left + (rcClientRect.right - rcGraph.left - (rcGraph.left - rcList.right)) / 2;
	else
		rcGraph.right = rcClientRect.right;
	m_ctrEpochGraphWnd.MoveWindow(rcGraph);

	rcLabel.left = rcGraph.left;
	rcLabel.right = rcGraph.right;
	m_ctrEpochGraphStatic.MoveWindow(rcLabel);

	if (m_has_second_history)
	{
		CRect rcTestGraph = rcGraph;
		rcTestGraph.left = rcGraph.right + (rcGraph.left - rcList.right) / 2;
		rcTestGraph.right = rcClientRect.right;
		m_ctrTestEpochGraphWnd.MoveWindow(rcTestGraph);

		rcLabel.left = rcTestGraph.left;
		rcLabel.right = rcTestGraph.right;
		m_ctrTestEpochGraphStatic.MoveWindow(rcLabel);
	}
}

void CAnalysisWnd::ReadySimulation(const _SIM_TRAIN_SETUP_INFO& sim_setup_info)
{
	Clear();
	m_has_argmax_accuracy = sim_setup_info.learn_info.analyze.bAnalyzeArgmaxAccuracy;
	m_has_second_history = sim_setup_info.learn_info.learn_type == engine::_learn_type::learn_test_both;

	m_ctrEpochHistoryList.DeleteAllItems();
	gui::win32::WinUtil::DeleteAllColumns(m_ctrEpochHistoryList);
	
	m_ctrTestEpochGraphWnd.ShowWindow(m_has_second_history ? SW_SHOWNORMAL : SW_HIDE);

	int column = m_ctrEpochHistoryList.InsertColumn(0, L"epoch");
	column = m_ctrEpochHistoryList.InsertColumn(column + 1, L"loss");
	if (m_has_argmax_accuracy)
		column = m_ctrEpochHistoryList.InsertColumn(column + 1, L"accuracy");
	if (m_has_second_history)
	{
		column = m_ctrEpochHistoryList.InsertColumn(column + 1, L"test loss");
		if (m_has_argmax_accuracy)
			column = m_ctrEpochHistoryList.InsertColumn(column + 1, L"test accuracy");
	}
	gui::win32::WinUtil::ResizeListControlHeader(m_ctrEpochHistoryList);

	m_ctrEpochGraphWnd.SetHasAccuracy(m_has_argmax_accuracy);
	m_ctrTestEpochGraphWnd.SetHasAccuracy(m_has_argmax_accuracy);
}

void CAnalysisWnd::Clear()
{
	m_ctrEpochHistoryList.DeleteAllItems();
	m_ctrEpochGraphWnd.Clear();
	m_ctrTestEpochGraphWnd.Clear();
}

void CAnalysisWnd::AddHistory(const _ANALYSIS_EPOCH_INFO& epoch_info)
{
	neuro_size_t epoch = m_ctrEpochHistoryList.GetItemCount();
	m_ctrEpochHistoryList.InsertItem(epoch, util::StringUtil::Transform<wchar_t>(epoch + 1).c_str());

	int row = 0;

	m_ctrEpochHistoryList.SetItemText(epoch, ++row, util::StringUtil::Transform<wchar_t>(epoch_info.epoch->loss).c_str());
	if (m_has_argmax_accuracy)
		m_ctrEpochHistoryList.SetItemText(epoch, ++row, util::StringUtil::Format(L"%3.2f %%", epoch_info.epoch->accuracy*neuron_value(100)).c_str());

	m_ctrEpochGraphWnd.AddData(epoch_info.epoch->loss, epoch_info.epoch->accuracy);

	if (m_has_second_history && epoch_info.test_epoch)
	{
		m_ctrEpochHistoryList.SetItemText(epoch, ++row, util::StringUtil::Transform<wchar_t>(epoch_info.test_epoch->loss).c_str());
		if (m_has_argmax_accuracy)
			m_ctrEpochHistoryList.SetItemText(epoch, ++row, util::StringUtil::Format(L"%3.2f %%", epoch_info.test_epoch->accuracy*neuron_value(100)).c_str());

		m_ctrTestEpochGraphWnd.AddData(epoch_info.test_epoch->loss, epoch_info.test_epoch->accuracy);
	}
	gui::win32::WinUtil::ResizeListControlHeader(m_ctrEpochHistoryList);
	m_ctrEpochHistoryList.EnsureVisible(epoch, FALSE);

	InvalidateRect(NULL, FALSE);
}
