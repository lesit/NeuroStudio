// SimRunDataDlg.cpp : 구현 파일입니다.
//

#include "stdafx.h"
#include "SimPredictSetupDlg.h"

#include "storage/MemoryDeviceAdaptor.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

CSimPredictSetupDlg::CSimPredictSetupDlg(np::project::NeuroStudioProject& project, SimulationRunningWnd& run_wnd)
: CAbstractSimSetupDlg(project, run_wnd)
, m_provider_model(project.GetNSManager().GetProvider().GetFinalProvider(true))
{
	m_nType = 0;
	if (m_provider_model.GetProducerVector().size() == 1)
	{
		dp::model::AbstractProducerModel* producer = m_provider_model.GetProducerVector()[0];
		dp::model::_input_source_type input_source_type = producer->GetInputSourceType(); 
		if (input_source_type == dp::model::_input_source_type::textfile)
			m_nType = 1;
		else if (input_source_type == dp::model::_input_source_type::imagefile)
			m_nType = 2;
	}

	const project::SimDefinition& sim_def = project.GetSimManager();

	const project::SIM_RUN_ENV& env = sim_def.GetRunEnv();
	m_display_period_batch = env.display_period_sample;

	m_strOutputFilePath = sim_def.GetLastPredictOutputPath().c_str();
	m_strOutputNoPrefix = sim_def.GetLastPredictOutputWriteInfo().no_type_prefix.c_str();

	m_minibatch_size = max(env.minibatch_size, 1);
}

CSimPredictSetupDlg::~CSimPredictSetupDlg()
{
}

void CSimPredictSetupDlg::DoDataExchange(CDataExchange* pDX)
{
	__super::DoDataExchange(pDX);
	DDX_Radio(pDX, IDC_RADIO_FILE, m_nType);
	DDX_Text(pDX, IDC_EDIT_OUTPUT_FILEPATH, m_strOutputFilePath);
	DDX_Text(pDX, IDC_EDIT_OUTPUT_NO_PREFIX, m_strOutputNoPrefix);
	DDX_Text(pDX, IDC_EDIT_BATCH_SIZE, m_minibatch_size);
	DDV_MinMaxUInt(pDX, m_minibatch_size, 1, neuro_last32);
}

static UINT msg_end_draw = gui::win32::PaintCtrl::GetEndDrawMessage();
BEGIN_MESSAGE_MAP(CSimPredictSetupDlg, CAbstractSimSetupDlg)
	ON_WM_SIZE()
	ON_BN_CLICKED(IDC_RADIO_FILE, OnChangedRadio)
	ON_BN_CLICKED(IDC_RADIO_TEXT, OnChangedRadio)
	ON_BN_CLICKED(IDC_RADIO_DRAW, OnChangedRadio)
	ON_REGISTERED_MESSAGE(msg_end_draw, OnEndDrawing)
END_MESSAGE_MAP()

BOOL CSimPredictSetupDlg::OnInitDialog()
{
	__super::OnInitDialog();

	m_ctrPaint.Create(NULL, NULL, WS_CHILD | WS_VISIBLE, CRect(), this, IDC_SIM_RUN_PAINT);
	m_ctrEdit.Create(ES_LEFT | ES_AUTOHSCROLL | WS_CHILD | WS_VISIBLE | WS_BORDER, CRect(), this, IDC_SIM_RUN_EDIT);

	std::vector<UINT> disable_vector;
	if (m_provider_model.GetProducerVector().size() > 1)
	{
		disable_vector = { IDC_RADIO_TEXT, IDC_RADIO_DRAW};
	}
	if (m_provider_model.GetProducerVector().size() == 1)
	{
		dp::model::_input_source_type input_source_type = m_provider_model.GetProducerVector()[0]->GetInputSourceType();

		if (input_source_type == dp::model::_input_source_type::textfile)
			disable_vector = { IDC_RADIO_DRAW };
		else if (input_source_type == dp::model::_input_source_type::imagefile)
			disable_vector = { IDC_RADIO_TEXT };
	}
	else
	{
		disable_vector = { IDC_RADIO_FILE, IDC_RADIO_TEXT, IDC_RADIO_DRAW };
	}

	for (size_t i = 0; i < disable_vector.size(); i++)
		GetDlgItem(disable_vector[i])->EnableWindow(FALSE);

	OnChangedRadio();

	InitDataTree();
	return TRUE;
}

void CSimPredictSetupDlg::InitDataTree()
{
	project::SimDefinition& sim_def = m_project.GetSimManager();

	m_ctrDataTreeWnd.m_ctrTree.DeleteAllItems();

	const dp::model::_producer_model_vector& producer_model_vector = m_provider_model.GetProducerVector();
	const dp::preprocessor::_uid_datanames_map& uid_path_map = sim_def.GetLastPredictData();

	m_ctrDataTreeWnd.m_ctrTree.InitProvider(producer_model_vector, uid_path_map);
}

void CSimPredictSetupDlg::OnChangedRadio()
{
	UpdateData(TRUE);

	m_ctrPaint.ShowWindow(SW_HIDE);
	m_ctrEdit.ShowWindow(SW_HIDE);
	m_ctrDataTreeWnd.ShowWindow(SW_HIDE);
	UINT show = 0;
	if (m_nType == 0)
		m_ctrDataTreeWnd.ShowWindow(SW_SHOWNORMAL);
	else if (m_nType == 1)
		m_ctrEdit.ShowWindow(SW_SHOWNORMAL);
	else
		m_ctrPaint.ShowWindow(SW_SHOWNORMAL);

	if (m_provider_model.GetProducerVector().size() == 0)
		m_ctrDataTreeWnd.EnableWindow(FALSE);
}

void CSimPredictSetupDlg::OnSize(UINT nType, int cx, int cy)
{
	CAbstractSimSetupDlg::OnSize(nType, cx, cy);

	if (GetDlgItem(IDC_DATA_GROUP) == NULL)
		return;

	CRect rcInputView;
	GetDlgItem(IDC_DATA_GROUP)->GetWindowRect(rcInputView);
	ScreenToClient(rcInputView);
	rcInputView.left += 5;
	rcInputView.right -= 5;

	CRect rc;
	if (m_provider_model.GetProducerVector().size() >= 2)
	{
		rcInputView.top += 20;
	}
	else
	{
		GetDlgItem(IDC_RADIO_FILE)->GetWindowRect(rc);
		ScreenToClient(rc);
		rcInputView.top = rc.bottom + 5;
	}

	GetDlgItem(IDC_STATIC_OUTPUT)->GetWindowRect(rc);
	ScreenToClient(rc);
	rcInputView.bottom = rc.top - 5;

	m_ctrPaint.MoveWindow(rcInputView);
	m_ctrDataTreeWnd.MoveWindow(rcInputView);
	m_ctrEdit.MoveWindow(rcInputView);

	m_ctrPaint.GetPaintControl().GetClientRect(rc);
	m_ctrPaint.GetPaintControl().NewCanvas(rc.Width(), rc.Height());
}

UINT CSimPredictSetupDlg::GetBottomChildWindowID() const
{
	return IDC_STATIC_DISPLAY_PERIOD;
}

void CSimPredictSetupDlg::GetAutoMovingChildArray(CUIntArray& idArray) const
{
	idArray.Add(IDC_STATIC_OUTPUT);
	idArray.Add(IDC_STATIC_FILE_PATH);
	idArray.Add(IDC_EDIT_OUTPUT_FILEPATH);
	idArray.Add(IDC_STATIC_NO_PREFIX);
	idArray.Add(IDC_EDIT_OUTPUT_NO_PREFIX);

	idArray.Add(IDC_STATIC_MINIBATCH_INFO);
	idArray.Add(IDC_STATIC_BATCH_SIZE);
	idArray.Add(IDC_EDIT_BATCH_SIZE);

	idArray.Add(IDC_STATIC_DISPLAY_PERIOD);
	idArray.Add(IDC_EDIT_DISPLAY_PERIOD);
}

void CSimPredictSetupDlg::GetAutoSizingChildArray(CUIntArray& idArray) const
{
	idArray.Add(IDC_DATA_GROUP);
}

#include "SimulationDlg.h"
LRESULT CSimPredictSetupDlg::OnEndDrawing(WPARAM wParam, LPARAM lParam)
{
	m_run_wnd.StartRunning();
	return 0L;
}

_SIM_SETUP_INFO* CSimPredictSetupDlg::CreateSetupInfo() const
{
	const_cast<CSimPredictSetupDlg*>(this)->UpdateData(TRUE);

	_SIM_PREDICT_SETUP_INFO* ret = new _SIM_PREDICT_SETUP_INFO;

	if (m_nType == 0)
	{
		m_ctrDataTreeWnd.m_ctrTree.GetDataFiles(ret->uid_datanames_map);
		if (ret->uid_datanames_map.size() == 0)
			goto failed;

		ret->predict_data_type = _SIM_PREDICT_SETUP_INFO::_predict_data_type::filepath;
	}
	else if(m_nType == 1)
	{
		CString str;
		m_ctrEdit.GetWindowText(str);

		ret->text_data = util::StringUtil::WideToMultiByte((const wchar_t*)str);

		ret->predict_data_type = _SIM_PREDICT_SETUP_INFO::_predict_data_type::text;
	}
	else
	{
		const dp::model::_producer_model_vector& producer_model_vector = m_provider_model.GetProducerVector();
		if (producer_model_vector.size() != 1)
			goto failed;

		tensor::DataShape shape = producer_model_vector[0]->GetDataShape();

		if (ret->img_data.Alloc(shape.GetDimSize()) == NULL)
			goto failed;

		if (!m_ctrPaint.GetPaintControl().ReadData(shape, -1.f, 1.f, true, ret->img_data.buffer))
			goto failed;

		ret->predict_data_type = _SIM_PREDICT_SETUP_INFO::_predict_data_type::image;
	}

	ret->minibatch_size = m_minibatch_size;

	ret->strOutputFilePath = util::StringUtil::WideToMultiByte((const wchar_t*)m_strOutputFilePath);

	return ret;

failed:
	delete ret;
	return NULL;
}

void CSimPredictSetupDlg::SaveConfig()
{
	UpdateData(TRUE);
	project::SimDefinition& sim_def = m_project.GetSimManager();

	project::SIM_RUN_ENV env;
	env.minibatch_size = m_minibatch_size;
	env.display_period_sample = m_display_period_batch;
	sim_def.SetRunEnv(env);

	sim_def.SetLastPredictOutputPath(m_strOutputFilePath);
	dp::_STREAM_WRITE_INFO& output_write_info = sim_def.GetLastPredictOutputWriteInfo();
	output_write_info.no_type_prefix = util::StringUtil::WideToMultiByte((const wchar_t*)m_strOutputNoPrefix);

	_uid_datanames_map& uid_datanames_map = sim_def.GetLastPredictData();
	m_ctrDataTreeWnd.m_ctrTree.GetDataFiles(uid_datanames_map);
}
