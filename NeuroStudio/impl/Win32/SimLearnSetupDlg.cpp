// SimTrainDataDlg.cpp : 구현 파일입니다.
//

#include "stdafx.h"
#include "SimLearnSetupDlg.h"
#include "util/FileUtil.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

CSimLearnSetupDlg::CSimLearnSetupDlg(np::project::NeuroStudioProject& project, SimulationRunningWnd& run_wnd)
	: CAbstractSimSetupDlg(project, run_wnd)
{
	const project::SIM_TRAIN_ENV& env = project.GetSimManager().GetTrainEnv();

	m_minibatch_size = max(env.minibatch_size, 1);
	m_max_epoch = env.max_epoch;
	m_is_stop_below_error = env.is_end_below_error;
	m_below_error = env.close_error;

	m_bTestAfterLearn = env.bTestAfterLearn;
	m_bAnalyzeArgmaxAccuracy = env.bAnalyzeArgmaxAccuracy;
	m_bAnalyzeLossHistory = env.bAnalyzeLossHistory;

	m_display_period_batch = env.display_period_sample;
	if (m_display_period_batch < 1)
		m_display_period_batch = 1;

	m_is_test = false;
}

CSimLearnSetupDlg::~CSimLearnSetupDlg()
{
}

void CSimLearnSetupDlg::DoDataExchange(CDataExchange* pDX)
{
	__super::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_EDIT_BATCH_SIZE, m_minibatch_size);
	DDV_MinMaxUInt(pDX, m_minibatch_size, 1, neuro_last32);
	DDX_Text(pDX, IDC_EDIT_MAX_EPOCH, m_max_epoch);
	DDV_MinMaxULongLong(pDX, m_max_epoch, 0, neuro_last64);
	DDX_Check(pDX, IDC_CHECK_BELOW_ERROR, m_is_stop_below_error);
	DDX_Text(pDX, IDC_EDIT_BELOW_ERROR, m_below_error);
	DDX_Check(pDX, IDC_CHECK_ANALYZE_ARGMAX_ACCURACY, m_bAnalyzeArgmaxAccuracy);
	DDX_Check(pDX, IDC_CHECK_ANALYZE_LOSS_HISTORY, m_bAnalyzeLossHistory);
	DDX_Check(pDX, IDC_CHECK_TEST_AFTER_LEARN, m_bTestAfterLearn);
}

BEGIN_MESSAGE_MAP(CSimLearnSetupDlg, CAbstractSimSetupDlg)
	ON_WM_SIZE()
	ON_BN_CLICKED(IDC_RADIO_TRAIN_LEARN, OnBnClickedRadioTrain)
	ON_BN_CLICKED(IDC_RADIO_TRAIN_TEST, OnBnClickedRadioTest)
END_MESSAGE_MAP()

BOOL CSimLearnSetupDlg::OnInitDialog()
{
	__super::OnInitDialog();

	CButton* pTrainBtn = (CButton*)GetDlgItem(IDC_RADIO_TRAIN_LEARN);
	if (!pTrainBtn)
		return FALSE;
	pTrainBtn->SetCheck(BST_CHECKED);

	if (m_project.GetSimManager().GetTrainEnv().useNdf)
		((CButton*)GetDlgItem(IDC_CHECK_USE_NDF))->SetCheck(BST_CHECKED);

	if (m_project.GetSimManager().GetTrainEnv().data_noising)
		((CButton*)GetDlgItem(IDC_CHECK_DATA_NOISING))->SetCheck(BST_CHECKED);

	InitDataTree();

	return TRUE;  // return TRUE unless you set the focus to a control
	// 예외: OCX 속성 페이지는 FALSE를 반환해야 합니다.
}

void CSimLearnSetupDlg::InitDataTree()
{
	project::SimDefinition& sim_def = m_project.GetSimManager();

	m_ctrDataTreeWnd.m_ctrTree.DeleteAllItems();

	const dp::model::_producer_model_vector& producer_model_vector = m_project.GetNSManager().GetProvider().GetFinalProvider(false).GetProducerVector();
	const dp::preprocessor::_uid_datanames_map& uid_path_map = m_is_test ? sim_def.GetLastTestData() : sim_def.GetLastLearnData();

	m_ctrDataTreeWnd.m_ctrTree.InitProvider(producer_model_vector, uid_path_map);
}

void CSimLearnSetupDlg::OnSize(UINT nType, int cx, int cy)
{
	__super::OnSize(nType, cx, cy);

	if (GetDlgItem(IDC_STATIC_DATA) == NULL)
		return;

	CRect rc;
	GetDlgItem(IDC_STATIC_DATA)->GetWindowRect(rc);
	ScreenToClient(rc);
	rc.left += 5;
	rc.right -= 5;

	CRect rcTemp;
	GetDlgItem(IDC_RADIO_TRAIN_LEARN)->GetWindowRect(rcTemp);
	ScreenToClient(rcTemp);
	rc.top = rcTemp.bottom + 10;

	GetDlgItem(IDC_CHECK_INIT_WEIGHTS)->GetWindowRect(rcTemp);
	ScreenToClient(rcTemp);
	rc.bottom = rcTemp.top - 5;

	m_ctrDataTreeWnd.MoveWindow(rc);
}

UINT CSimLearnSetupDlg::GetBottomChildWindowID() const
{
	return IDC_EDIT_DISPLAY_PERIOD;
}

void CSimLearnSetupDlg::GetAutoMovingChildArray(CUIntArray& idArray) const
{
	idArray.Add(IDC_CHECK_INIT_WEIGHTS);
	idArray.Add(IDC_CHECK_USE_NDF);
	idArray.Add(IDC_CHECK_DATA_NOISING);

	idArray.Add(IDC_STATIC_MINIBATCH_INFO);
	idArray.Add(IDC_STATIC_BATCH_SIZE);
	idArray.Add(IDC_EDIT_BATCH_SIZE);

	idArray.Add(IDC_STATIC_END_CONDITION);
	idArray.Add(IDC_STATIC_MAX_EPOCH);
	idArray.Add(IDC_EDIT_MAX_EPOCH);
	idArray.Add(IDC_CHECK_BELOW_ERROR);
	idArray.Add(IDC_EDIT_BELOW_ERROR);
	idArray.Add(IDC_STATIC_ANALYSYS);
	idArray.Add(IDC_CHECK_TEST_AFTER_LEARN);
	idArray.Add(IDC_CHECK_ANALYZE_ARGMAX_ACCURACY);
	idArray.Add(IDC_CHECK_ANALYZE_LOSS_HISTORY);
	idArray.Add(IDC_STATIC_DISPLAY);
}

void CSimLearnSetupDlg::GetAutoSizingChildArray(CUIntArray& idArray) const
{
	idArray.Add(IDC_STATIC_DATA);
}

void CSimLearnSetupDlg::OnBnClickedRadioTrain()
{
	if (m_is_test)	// 변화가 있을 때만 하자!
	{
		SaveConfig();

		m_is_test = false;
		InitDataTree();

		SetWndCtrlStatus();
	}
}

void CSimLearnSetupDlg::OnBnClickedRadioTest()
{
	if (!m_is_test)
	{
		SaveConfig();

		m_is_test = true;
		InitDataTree();

		SetWndCtrlStatus();
	}
}

void CSimLearnSetupDlg::SetWndCtrlStatus()
{
	if (m_is_test)
	{
		UINT disable_ctrl_array[] = { IDC_EDIT_MAX_EPOCH, IDC_EDIT_BELOW_ERROR, IDC_CHECK_BELOW_ERROR, IDC_CHECK_INIT_WEIGHTS, IDC_CHECK_DATA_NOISING };
		for (int i = 0; i < _countof(disable_ctrl_array); i++)
			GetDlgItem(disable_ctrl_array[i])->EnableWindow(FALSE);
	}
	else
	{
		UINT enable_ctrl_array[] = { IDC_EDIT_MAX_EPOCH, IDC_EDIT_BELOW_ERROR, IDC_CHECK_BELOW_ERROR, IDC_CHECK_INIT_WEIGHTS, IDC_CHECK_DATA_NOISING };
		for (int i = 0; i < _countof(enable_ctrl_array); i++)
			GetDlgItem(enable_ctrl_array[i])->EnableWindow(TRUE);
	}
}

_SIM_SETUP_INFO* CSimLearnSetupDlg::CreateSetupInfo() const
{
	const_cast<CSimLearnSetupDlg*>(this)->UpdateData(TRUE);

	_SIM_TRAIN_SETUP_INFO* ret = new _SIM_TRAIN_SETUP_INFO;

	m_ctrDataTreeWnd.m_ctrTree.GetDataFiles(ret->uid_datanames_map);
	if (ret->uid_datanames_map.size() == 0)
	{
		delete ret;
		return NULL;
	}

	const _uid_datanames_map& test_data_source_name_map = m_project.GetSimManager().GetLastTestData();

	if (m_is_test)
	{
		ret->learn_info.learn_type = engine::_learn_type::test;
	}
	else
	{
		ret->learn_info.learn_type = engine::_learn_type::learn;
		if (m_bTestAfterLearn && test_data_source_name_map.size()>0)
		{
			bool has_full_data = true;
			_uid_datanames_map::const_iterator it = test_data_source_name_map.begin();
			for (; it != test_data_source_name_map.end(); it++)
			{
				if (it->second.GetCount() == 0)
				{
					has_full_data = false;
					break;
				}
			}
			if (has_full_data)
			{
				ret->learn_info.learn_type = engine::_learn_type::learn_test_both;
				ret->test_uid_datanames_map = test_data_source_name_map;
			}
		}
	}

	ret->useNdf = ((CButton*)GetDlgItem(IDC_CHECK_USE_NDF))->GetCheck() == BST_CHECKED;
	ret->data_noising = !m_is_test && ((CButton*)GetDlgItem(IDC_CHECK_DATA_NOISING))->GetCheck() == BST_CHECKED;

	ret->learn_info.isWeightInit = ((CButton*)GetDlgItem(IDC_CHECK_INIT_WEIGHTS))->GetCheck() == BST_CHECKED;

	ret->learn_info.analyze.bAnalyzeArgmaxAccuracy = m_bAnalyzeArgmaxAccuracy != FALSE;
	ret->learn_info.analyze.bAnalyzeLossHistory = m_bAnalyzeLossHistory != FALSE;

	ret->learn_info.epoch_count = m_max_epoch < 1 ? 1 : m_max_epoch;
	ret->minibatch_size = m_minibatch_size;

//	ret->max_epoch = m_max_epoch;
	ret->is_stop_below_error = m_is_stop_below_error!=FALSE;
	ret->below_error = m_below_error;
	return ret;
}

void CSimLearnSetupDlg::BeforeRun()
{
	UINT disable_ctrl_array[] = { IDC_RADIO_TRAIN_LEARN, IDC_RADIO_TRAIN_TEST, IDC_EDIT_MAX_EPOCH, IDC_EDIT_BATCH_SIZE, IDC_EDIT_BELOW_ERROR, IDC_CHECK_BELOW_ERROR, IDC_CHECK_ANALYZE_ARGMAX_ACCURACY, IDC_CHECK_ANALYZE_LOSS_HISTORY, IDC_CHECK_TEST_AFTER_LEARN, IDC_CHECK_INIT_WEIGHTS, IDC_CHECK_USE_NDF, IDC_CHECK_DATA_NOISING };
	for (int i = 0; i < _countof(disable_ctrl_array); i++)
		GetDlgItem(disable_ctrl_array[i])->EnableWindow(FALSE);
}

void CSimLearnSetupDlg::AfterRun()
{
	UINT enable_ctrl_array[] = { IDC_RADIO_TRAIN_LEARN, IDC_RADIO_TRAIN_TEST, IDC_EDIT_MAX_EPOCH, IDC_EDIT_BATCH_SIZE, IDC_EDIT_BELOW_ERROR, IDC_CHECK_BELOW_ERROR, IDC_CHECK_ANALYZE_ARGMAX_ACCURACY, IDC_CHECK_ANALYZE_LOSS_HISTORY, IDC_CHECK_TEST_AFTER_LEARN, IDC_CHECK_INIT_WEIGHTS, IDC_CHECK_USE_NDF, IDC_CHECK_DATA_NOISING };
	for (int i = 0; i < _countof(enable_ctrl_array); i++)
		GetDlgItem(enable_ctrl_array[i])->EnableWindow(TRUE);

	((CButton*)GetDlgItem(IDC_CHECK_INIT_WEIGHTS))->SetCheck(BST_UNCHECKED);

	SetWndCtrlStatus();
}

void CSimLearnSetupDlg::SaveConfig()
{
	UpdateData(TRUE);
	project::SimDefinition& sim_def = m_project.GetSimManager();

	project::SIM_TRAIN_ENV env;
	env.useNdf = ((CButton*)GetDlgItem(IDC_CHECK_USE_NDF))->GetCheck() != BST_UNCHECKED;
	env.data_noising = ((CButton*)GetDlgItem(IDC_CHECK_DATA_NOISING))->GetCheck() == BST_CHECKED;

	env.minibatch_size = m_minibatch_size;
	env.max_epoch = m_max_epoch;
	env.is_end_below_error = m_is_stop_below_error!=FALSE;
	env.close_error = m_below_error;

	env.bTestAfterLearn = m_bTestAfterLearn != FALSE;
	env.bAnalyzeArgmaxAccuracy = m_bAnalyzeArgmaxAccuracy != FALSE;
	env.bAnalyzeLossHistory = m_bAnalyzeLossHistory != FALSE;

	env.display_period_sample = m_display_period_batch;
	sim_def.SetTrainEnv(env);

	_uid_datanames_map& uid_datanames_map = m_is_test ? sim_def.GetLastTestData() : sim_def.GetLastLearnData();
	m_ctrDataTreeWnd.m_ctrTree.GetDataFiles(uid_datanames_map);
}

void CSimLearnSetupDlg::ViewAnalysisHistory()
{
//	CAnalysisDlg dlg(m_analysis_history, this);
//	dlg.DoModal();
}
