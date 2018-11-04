#include "stdafx.h"

#include "SimulationRunningWnd.h"
#include "util/FileUtil.h"

#include "SimulationDlg.h"

SimulationRunningWnd::SimulationRunningWnd(CSimulationDlg& simDlg)
	: m_simDlg(simDlg)
	, m_layerDisplayWnd(m_layer_display_item_matrix_vector)
	, m_init_share_object(this, util::FileUtil::GetDirFromPath<char>(simDlg.GetProject().GetProjectFilePath()))
{
	m_hIntNetworkThread = NULL;
	m_hRunThread = NULL;

	m_pdType =  (m_p_instance.cuda_instance ? core::math_device_type::cuda : core::math_device_type::cpu);

	m_network = NULL;

	m_has_analysis = false;
	m_has_onehot_result = false;

	m_hAwakeFromPause = CreateEvent(NULL, TRUE, FALSE, NULL);

	m_backBrush.CreateSolidBrush(RGB(255, 255, 255));
}

SimulationRunningWnd::~SimulationRunningWnd()
{
	CloseHandle(m_hAwakeFromPause);

	for (neuro_u32 level = 0, level_count = m_layer_display_item_matrix_vector.size(); level < level_count; level++)
	{
		_layer_display_item_rowl_vector& row_vector = m_layer_display_item_matrix_vector[level];
		for (neuro_u32 row = 0; row < row_vector.size(); row++)
		{
			LayerDisplayItem& item = row_vector[row];
			item.buffer.output.Dealloc();
			item.buffer.target.Dealloc();
		}
	}

	m_sim_instance.Clear();

	if (m_network)
		delete m_network;

}

UINT NPM_INITIALIZED = ::RegisterWindowMessage(_T("NP.Impl.Sim.Initialized"));
UINT NPM_SIM_NET_SIGNAL = ::RegisterWindowMessage(_T("NP.Impl.Sim.NetSignal"));
UINT NPM_SIM_MESSAGE = ::RegisterWindowMessage(_T("NP.Impl.Sim.Message"));
UINT NPM_SIM_REDRAW_RESULT = ::RegisterWindowMessage(_T("NP.Impl.Sim.RedrawResult"));
UINT NPM_COMPLETED_RUN = ::RegisterWindowMessage(_T("NP.Impl.Sim.Completed"));

BEGIN_MESSAGE_MAP(SimulationRunningWnd, CWnd)
	ON_WM_CREATE()
	ON_BN_CLICKED(IDC_RADIO_PARALLEL_CUDA, OnBnClickedRadioParallelType)
	ON_BN_CLICKED(IDC_RADIO_PARALLEL_CPU, OnBnClickedRadioParallelType)
	ON_BN_CLICKED(IDC_BUTTON_START, OnBnClickedStart)
	ON_BN_CLICKED(IDC_BUTTON_STOP, OnBnClickedEnd)
	ON_BN_CLICKED(IDC_BUTTON_STOP_EPOCH, OnBnClickedButtonStopEpoch)
	ON_REGISTERED_MESSAGE(NPM_INITIALIZED, OnInitializedMessage)
	ON_REGISTERED_MESSAGE(NPM_SIM_NET_SIGNAL, OnSimNetSignal)
	ON_REGISTERED_MESSAGE(NPM_SIM_MESSAGE, OnSimMessage)
	ON_REGISTERED_MESSAGE(NPM_SIM_REDRAW_RESULT, OnRedrawResults)
	ON_REGISTERED_MESSAGE(NPM_COMPLETED_RUN, OnCompletedRunMessage)
	ON_WM_SIZE()
	ON_WM_CTLCOLOR()
	ON_WM_PAINT()
END_MESSAGE_MAP()

int SimulationRunningWnd::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (__super::OnCreate(lpCreateStruct) == -1)
		return -1;

	const DWORD dwDefaultStyle = WS_CHILD;// | WS_CLIPSIBLINGS | WS_CLIPCHILDREN;

	CRect rcDummy(0, 0, 0, 0);
	m_ctrPdTypeStatic.Create(L"Processor", dwDefaultStyle | WS_VISIBLE, rcDummy, this);

	DWORD dwRadioStyle = dwDefaultStyle | WS_VISIBLE | BS_AUTORADIOBUTTON | WS_TABSTOP;
	m_ctrGpuRadioBtn.Create(L"GPU (Nvida)", dwRadioStyle | WS_GROUP, rcDummy, this, IDC_RADIO_PARALLEL_CUDA);
	m_ctrCpuRadioBtn.Create(L"CPU", dwRadioStyle, rcDummy, this, IDC_RADIO_PARALLEL_CPU);

	m_ctrStartBtn.Create(L"Start", dwDefaultStyle | WS_VISIBLE | WS_TABSTOP | BS_DEFPUSHBUTTON, rcDummy, this, IDC_BUTTON_START);
	m_ctrStopBtn.Create(L"Stop", dwDefaultStyle | WS_VISIBLE | WS_TABSTOP, rcDummy, this, IDC_BUTTON_STOP);
	m_ctrStopEpochBtn.Create(L"Stop epoch", dwDefaultStyle | WS_TABSTOP, rcDummy, this, IDC_BUTTON_STOP_EPOCH);

	m_ctrElapseLabelStatic.Create(L"Elapse", dwDefaultStyle | WS_VISIBLE | SS_CENTERIMAGE | SS_RIGHT, rcDummy, this);
	m_ctrElapseStatic.Create(L"", dwDefaultStyle | WS_VISIBLE | WS_BORDER | SS_CENTERIMAGE, rcDummy, this);

	m_ctrRunStatusStatic.Create(L"ready", dwDefaultStyle | WS_VISIBLE | WS_BORDER | SS_CENTERIMAGE, rcDummy, this);

	m_ctrStatusGrid.Create(dwDefaultStyle | WS_VISIBLE | WS_BORDER, rcDummy, this, IDC_PROP_LIST);
	m_ctrStatusGrid.EnableHeaderCtrl(FALSE);
	m_ctrStatusGrid.SetVSDotNetLook();

	m_analysisWnd.Create(NULL, NULL, dwDefaultStyle | WS_VISIBLE, rcDummy, this, IDC_ANALYSIS);

	m_layerDisplayWnd.Create(NULL, NULL, dwDefaultStyle | WS_VISIBLE | WS_BORDER, rcDummy, this, IDC_LAYER_DISPLAY);

	m_ctrOnehotResultStatic.Create(L"One hot result", dwDefaultStyle, rcDummy, this);
	// CGroupListCtrl의 PreSubclassWindow() 확인해야함
	m_ctrOnehotResultList.Create(dwDefaultStyle | WS_BORDER | LVS_REPORT | LVS_SINGLESEL | LVS_SHOWSELALWAYS, rcDummy, this, IDC_ONEHOT_RESULT_LIST);	

	if (m_pdType == core::math_device_type::cuda)
		m_ctrGpuRadioBtn.SetCheck(BST_CHECKED);
	else
		m_ctrCpuRadioBtn.SetCheck(BST_CHECKED);

	m_ctrGpuRadioBtn.EnableWindow(m_p_instance.cuda_instance != NULL);

	UINT disable_ctrl_array[] = { IDC_BUTTON_START, IDC_BUTTON_STOP };
	for (int i = 0; i < _countof(disable_ctrl_array); i++)
		GetDlgItem(disable_ctrl_array[i])->EnableWindow(FALSE);

	m_sim_instance.display_period_batch = m_simDlg.GetDisplayPeriodBatch();

	// TODO:  Add your specialized creation code here
	DWORD dwThreadId;
	m_hIntNetworkThread = CreateThread(NULL, 0, IntNetworkThread, this, CREATE_SUSPENDED, &dwThreadId);
	ResumeThread(m_hIntNetworkThread);

	return 0;
}

DWORD WINAPI SimulationRunningWnd::IntNetworkThread(LPVOID pParam)
{
	DEBUG_OUTPUT(_T("start\r\n"));

	SimulationRunningWnd* _this = (SimulationRunningWnd*)pParam;

	UINT disable_ctrl_array[] = { IDC_BUTTON_START, IDC_RADIO_PARALLEL_CUDA, IDC_RADIO_PARALLEL_CPU };
	for (int i = 0; i < _countof(disable_ctrl_array); i++)
		_this->GetDlgItem(disable_ctrl_array[i])->EnableWindow(FALSE);

	_this->m_ctrRunStatusStatic.SetWindowText(L"loading network");

	bool ret = false;
	if (!_this->CreateNetworkInstance((core::math_device_type)_this->m_pdType))
	{
		DEBUG_OUTPUT(_this->m_strInitFailedMsg);
		_this->m_ctrRunStatusStatic.SetWindowText(_this->m_strInitFailedMsg);
		goto end;
	}
	_this->m_ctrRunStatusStatic.SetWindowText(L"loading network is completed");

	ret = true;

	_this->m_ctrStartBtn.EnableWindow(TRUE);
	_this->m_ctrGpuRadioBtn.EnableWindow(_this->m_p_instance.cuda_instance != NULL);
	_this->m_ctrCpuRadioBtn.EnableWindow(TRUE);

end:
	thread::Lock::Owner lock(_this->m_initThreadLock);
	_this->m_hIntNetworkThread = NULL;

	_this->PostMessage(NPM_INITIALIZED, (WPARAM)ret);
	DEBUG_OUTPUT(_T("end\r\n"));
	return 0;
}

HBRUSH SimulationRunningWnd::OnCtlColor(CDC* pDC, CWnd* pWnd, UINT nCtlColor)
{
	HBRUSH hbr = __super::OnCtlColor(pDC, pWnd, nCtlColor);

	if (nCtlColor == CTLCOLOR_STATIC)
	{
		pDC->SetBkColor(RGB(255, 255, 255));
		return m_backBrush;
	}

	return hbr;
}

void SimulationRunningWnd::OnPaint()
{
	Invalidate();	// 이렇게 해야 실행도중 왔다갔다 해도 status grid도 boder도 보인다.

	CPaintDC paintDC(this); // device context for painting
							// TODO: Add your message handler code here
							// Do not call __super::OnPaint() for painting messages
	CRect rcClient;
	GetClientRect(&rcClient);	// 전체 영역을 얻는다.

	CMemDC memDC(paintDC, rcClient);
	CDC& dc = memDC.GetDC();
	dc.FillSolidRect(&rcClient, RGB(255, 255, 255));
}

void SimulationRunningWnd::OnSize(UINT nType, int cx, int cy)
{
	__super::OnSize(nType, cx, cy);

	if (m_ctrPdTypeStatic.GetSafeHwnd() == NULL)
		return;

	CRect rcClient;
	GetClientRect(&rcClient);
	rcClient.DeflateRect(5, 5);

	CRect rc;
	rc.left = rcClient.left;
	rc.right = 150;
	rc.top = rcClient.top;
	rc.bottom = rcClient.top + 30;
	m_ctrPdTypeStatic.MoveWindow(&rc);

	rc.left = rc.right + 5;
	rc.right = rc.left + 100;
	m_ctrGpuRadioBtn.MoveWindow(&rc);

	rc.left = rc.right + 5;
	rc.right = rc.left + 80;
	m_ctrCpuRadioBtn.MoveWindow(&rc);

	rc.left = rc.right + 20;
	rc.right = rc.left + 80;
	m_ctrStartBtn.MoveWindow(&rc);
	rc.left = rc.right + 5;
	rc.right = rc.left + 80;
	m_ctrStopBtn.MoveWindow(&rc);
	rc.left = rc.right + 5;
	rc.right = rc.left + 120;
	m_ctrStopEpochBtn.MoveWindow(&rc);

	rc.right = rcClient.right;
	rc.left = rc.right - 120;
	m_ctrElapseStatic.MoveWindow(&rc);
	rc.right = rc.left - 5;
	rc.left = rc.right - 70;
	m_ctrElapseLabelStatic.MoveWindow(&rc);

	rc.top = rc.bottom + 10;
	rc.bottom = rc.top + 25;
	rc.left = 5;
	rc.right = rcClient.right;
	m_ctrRunStatusStatic.MoveWindow(&rc);

	rc.top = rc.bottom + 10;
	rc.bottom = rc.top + (rcClient.Height() - 5 - rc.top - 10) / 2;

	rc.left = rcClient.left;
	rc.right = rc.left + 270;// max(270, rcClient.Width() / 4);
	m_ctrStatusGrid.MoveWindow(rc);

	if (m_has_analysis)
	{
		rc.left = rc.right + 10;
		rc.right = rcClient.right;
		m_analysisWnd.MoveWindow(rc);
	}

	CRect rcDesignWnd;
	rcDesignWnd.top = rc.bottom + 10;
	rcDesignWnd.bottom = rcClient.bottom;
	rcDesignWnd.left = rcClient.left;
	rcDesignWnd.right = rcClient.right;
	if (m_has_onehot_result)
	{
		rc = rcDesignWnd;
		rc.left = rcDesignWnd.right - 200;
		rc.right = rcDesignWnd.right;
		rc.bottom = rc.top + 25;
		m_ctrOnehotResultStatic.MoveWindow(rc);

		rc.top = rc.bottom + 5;
		rc.bottom = rcDesignWnd.bottom;
		m_ctrOnehotResultList.MoveWindow(rc);

		rcDesignWnd.right = rc.left - 10;
	}
	m_layerDisplayWnd.MoveWindow(rcDesignWnd);
}

void SimulationRunningWnd::SimTypeChanged()
{
//	m_ctrStopEpochBtn.ShowWindow(m_simDlg.GetSimType() == np::simulate::_sim_type::train ? SW_SHOW : SW_HIDE);

//	ClearResult();
}

bool SimulationRunningWnd::IsCompletedInitNetwork()
{
	thread::Lock::Owner lock(m_initThreadLock);

	return m_hIntNetworkThread == NULL;
}

LRESULT SimulationRunningWnd::OnInitializedMessage(WPARAM wParam, LPARAM lParam)
{
	if (wParam == 0)
		MessageBox(m_strInitFailedMsg, L"Simulator");
	else
		DisplaySetupChanged();

	return 0;
}

bool SimulationRunningWnd::CreateNetworkInstance(core::math_device_type pd_type)
{
	DEBUG_OUTPUT(L"start");

	if (pd_type == core::math_device_type::cuda && !m_p_instance.cuda_instance)
	{
		m_strInitFailedMsg = L"no cuda device";
		return false;
	}

	network::NeuralNetwork* network_model = m_simDlg.GetProject().GetNSManager().GetNetwork();
	if (!network_model)
	{
		m_strInitFailedMsg = L"no network network";
		return false;
	}

	delete m_network;
	m_network = engine::NeuralNetworkEngine::CreateInstance(pd_type, m_p_instance,  *network_model);
	if (!m_network)
	{
		m_strInitFailedMsg = L"failed create instance";
		return false;
	}

	return true;
}

void SimulationRunningWnd::OnBnClickedRadioParallelType()
{
	core::math_device_type prev_pdType = m_pdType;

	m_pdType = m_ctrGpuRadioBtn.GetCheck() == BST_CHECKED ? core::math_device_type::cuda : core::math_device_type::cpu;

	if (m_pdType == prev_pdType)
		return;

	if (!IsCompletedInitNetwork())
	{
		DEBUG_OUTPUT(L"wait for init network. start");
		WaitForSingleObject(m_hIntNetworkThread, INFINITE);
		DEBUG_OUTPUT(L"wait for init network. end");
	}

	DWORD dwThreadId;
	m_hIntNetworkThread = CreateThread(NULL, 0, IntNetworkThread, this, CREATE_SUSPENDED, &dwThreadId);
	ResumeThread(m_hIntNetworkThread);
}

void SimulationRunningWnd::ClearResult()
{
	for (neuro_u32 level = 0, level_count = m_layer_display_item_matrix_vector.size(); level < level_count; level++)
	{
		_layer_display_item_rowl_vector& row_vector = m_layer_display_item_matrix_vector[level];
		for (neuro_u32 row = 0; row < row_vector.size(); row++)
		{
			LayerDisplayItem& item = row_vector[row];
			item.buffer.output.SetZero();
			item.buffer.target.SetZero();
		}
	}

	m_ctrStatusGrid.RemoveAll();

	m_analysisWnd.Clear();
}

void SimulationRunningWnd::DisplaySetupChanged()
{
	if (!m_network)
		return;

	thread::Lock::Owner lock(m_layer_display_lock);

	const engine::_uid_engine_map& engine_map = m_network->GetUidEngineMap();

	m_onehot_result_output_set.clear();

	// 기존 버퍼에서 사용할것은 재사용하기 위해서 미리 가지고 있는다.
	std::unordered_map<neuro_u32, _LAYER_OUT_BUF> prev_buffer_map;
	for (neuro_u32 level = 0, level_count = m_layer_display_item_matrix_vector.size(); level < level_count; level++)
	{
		const _layer_display_item_rowl_vector& row_vector = m_layer_display_item_matrix_vector[level];
		for (neuro_u32 row = 0; row < row_vector.size(); row++)
		{
			const LayerDisplayItem& item = row_vector[row];
			prev_buffer_map[item.layer_uid] = item.buffer;
		}
	}
	const _layer_display_setup_matrix_vector& setup_matrix_vector = m_simDlg.GetDisplayInfo();
	m_layer_display_item_matrix_vector.resize(setup_matrix_vector.size());
	for (neuro_u32 level = 0, level_count = setup_matrix_vector.size(); level < level_count; level++)
	{
		const _layer_display_setup_row_vector& setup_row_vector = setup_matrix_vector[level];
		_layer_display_item_rowl_vector& display_row_vector = m_layer_display_item_matrix_vector[level];
		display_row_vector.resize(setup_row_vector.size());
		for (neuro_u32 row = 0; row < setup_row_vector.size(); row++)
		{
			const LayerDisplaySetup& setup = setup_row_vector[row];
			const neuro_u32 uid = setup.layer->uid;

			engine::_uid_engine_map::const_iterator it_engine = engine_map.find(uid);
			if (it_engine == engine_map.end())
			{
				DEBUG_OUTPUT(L"no layer engine! layer[%u, %u]", setup.mp.level, setup.mp.row);
				m_layerDisplayWnd.RedrawResults();
			}
			LayerDisplayItem& item = display_row_vector[row];
			item.layer_uid = it_engine->second->m_layer.uid;
			item.engine = it_engine->second;
			item.mp = setup.mp;
			item.display = setup.display;

			std::unordered_map<neuro_u32, _LAYER_OUT_BUF>::const_iterator it_buf = prev_buffer_map.find(uid);
			if (it_buf == prev_buffer_map.end())
			{
				// m_layer_display_item_matrix_vector를 clear하지 않고 재사용하기 때문에 여기에서 초기화해야 한다.
				item.buffer = _LAYER_OUT_BUF();

				_LAYER_SCALE_INFO info;
				if (m_network->GetLayerOutInfo(uid, info))
				{
					item.buffer.low_scale = info.low_scale;
					item.buffer.up_scale = info.up_scale;
				}
			}
			else
			{
				item.buffer = it_buf->second;
				prev_buffer_map.erase(it_buf);
			}

			if (item.display.is_onehot_analysis_result)
				m_onehot_result_output_set.insert(uid);
		}
	}

	// 제외된 layer의 버퍼는 해제한다.
	std::unordered_map<neuro_u32, _LAYER_OUT_BUF>::iterator it_prev = prev_buffer_map.begin();
	for (; it_prev != prev_buffer_map.end(); it_prev++)
	{
		_LAYER_OUT_BUF& buf = it_prev->second;
		buf.output.Dealloc();
		buf.target.Dealloc();
	}
	m_layerDisplayWnd.RedrawResults();
}

void SimulationRunningWnd::PatchResults()
{
	if (!m_sim_instance.simulator)
	{
		DEBUG_OUTPUT(L"no simulator");
		return;
	}

	thread::Lock::Owner lock(m_layer_display_lock);

	const bool is_learn_mode = m_sim_instance.simulator->GetType() == simulate::_sim_type::train;
	for (neuro_u32 level = 0, level_count = m_layer_display_item_matrix_vector.size(); level < level_count; level++)
	{
		_layer_display_item_rowl_vector& row_vector = m_layer_display_item_matrix_vector[level];
		for (neuro_u32 row = 0; row < row_vector.size(); row++)
		{
			LayerDisplayItem& item = row_vector[row];

			if (!item.buffer.output.AllocCopyFrom(item.engine->GetOutputData()))
				continue;

			if (is_learn_mode && item.engine->GetLayerType()==network::_layer_type::output)
			{
				if (!item.buffer.target.AllocCopyFrom(((const OutputLayerEngine*)item.engine)->GetTargetData()))
				{
					DEBUG_OUTPUT(L"failed copy target data. layer[%u, %u]", item.mp.level, item.mp.row);
				}
			}
		}
	}

	PostMessage(NPM_SIM_REDRAW_RESULT);
}

void SimulationRunningWnd::StartRunning()
{
	OnBnClickedStart();
}

void SimulationRunningWnd::ReadySimulation(const _SIM_SETUP_INFO& sim_setup_info)
{
	m_sim_instance.sim_control_type = _sim_control_type::run;
	m_sim_instance.display_period_batch = m_simDlg.GetDisplayPeriodBatch();

	UINT disable_ctrl_array[] = { IDC_RADIO_PARALLEL_CUDA, IDC_RADIO_PARALLEL_CPU };
	for (int i = 0; i < _countof(disable_ctrl_array); i++)
		GetDlgItem(disable_ctrl_array[i])->EnableWindow(FALSE);

	m_ctrStartBtn.SetWindowText(L"Pause");
	m_ctrStopBtn.EnableWindow(TRUE);

	if (m_simDlg.GetSimType() == np::simulate::_sim_type::train)
	{
		m_ctrStopEpochBtn.ShowWindow(SW_SHOWNORMAL);
		m_ctrStopEpochBtn.EnableWindow(TRUE);
		m_ctrStopEpochBtn.SetWindowText(L"Stop Epoch");
	}
	else
	{
		m_ctrStopEpochBtn.ShowWindow(SW_HIDE);
	}

	m_simDlg.ReadySimulation();

	m_has_analysis = false;
	m_has_onehot_result = false;
	if (sim_setup_info.GetRunType() == np::simulate::_sim_type::train)
	{
		if (((_SIM_TRAIN_SETUP_INFO&)sim_setup_info).learn_info.learn_type != engine::_learn_type::test)
			m_has_analysis = true;

		// onehot result를 가지는 display가 있을 경우에만
		const np::project::_layer_display_info_map&  layer_display_map = m_simDlg.GetProject().GetSimManager().GetLayerDisplayInfoMap();
		_layer_display_info_map::const_iterator it = layer_display_map.begin();
		for (; it != layer_display_map.end(); it++)
		{
			if(it->second.is_onehot_analysis_result)
			{
				m_has_onehot_result = true;
				break;
			}
		}
	}

	m_analysisWnd.ShowWindow(m_has_analysis ? SW_SHOWNORMAL : SW_HIDE);
	m_ctrOnehotResultStatic.ShowWindow(m_has_onehot_result ? SW_SHOWNORMAL : SW_HIDE);
	m_ctrOnehotResultList.ShowWindow(m_has_onehot_result ? SW_SHOWNORMAL : SW_HIDE);

	SendMessage(WM_SIZE);
}

void SimulationRunningWnd::EndSimulation()
{
	m_sim_instance.Clear();

	m_simDlg.EndSimulation();

	m_ctrGpuRadioBtn.EnableWindow(m_p_instance.cuda_instance != NULL);
	m_ctrCpuRadioBtn.EnableWindow(TRUE);

	m_ctrStartBtn.SetWindowText(L"Start");
	m_ctrStartBtn.EnableWindow(TRUE);
	m_ctrStopBtn.EnableWindow(FALSE);
	m_ctrStopEpochBtn.ShowWindow(SW_HIDE);
	m_ctrStopEpochBtn.MoveWindow(0, 0, 0, 0);
	m_ctrStopEpochBtn.InvalidateRect(NULL, TRUE);
	InvalidateRect(NULL, TRUE);
	UpdateWindow();
	SendMessage(WM_SIZE);
}

void SimulationRunningWnd::OnBnClickedStart()
{
	if (m_hRunThread)
	{
		if (m_sim_instance.sim_control_type != _sim_control_type::pause)
		{
			DEBUG_OUTPUT(L"pause...");
			m_sim_instance.sim_control_type = _sim_control_type::pause;
			m_ctrStopEpochBtn.EnableWindow(FALSE);
		}
		else
		{
			m_sim_instance.sim_control_type = _sim_control_type::run;
			DEBUG_OUTPUT(L"awake from pause...");
			SetEvent(m_hAwakeFromPause);
			m_ctrStopEpochBtn.EnableWindow(TRUE);
		}
	}
	else
	{
		if (!IsCompletedInitNetwork())
		{
			if (MessageBox(L"Not yet completed an initialize of network\r\n\r\nDo you want wait ?", L"Simulation", MB_YESNO)
				== IDNO)
				return;

			if (!IsCompletedInitNetwork())
			{
				WaitForSingleObject(m_hIntNetworkThread, 1000 * 5);	// 5초만 기다려보자

				if (!IsCompletedInitNetwork())
				{
					MessageBox(L"Not yet completed an initialize of network\r\n\r\nTry again later", L"Simulation", MB_OK);
					return;
				}
			}
		}

		if (m_network == NULL)
		{
			CString strStatus;
			strStatus.Format(L"failed initialize network for %s parallel type", m_pdType == core::math_device_type::cuda ? L"GPU" : L"CPU");
			MessageBox(strStatus, L"Simulation");
			return;
		}

		_SIM_SETUP_INFO* sim_setup_info = m_simDlg.CreateSetupInfo();
		if (sim_setup_info == NULL)
		{
			CString strStatus;
			strStatus.Format(L"failed initialize setup info");
			MessageBox(strStatus, L"Simulation");
			return;
		}

		ClearResult();

		m_sim_instance.Clear();

		if (sim_setup_info->GetRunType() == simulate::_sim_type::train)
			m_sim_instance.sim_control = new SimulationTrainStatusControl(*this, (const _SIM_TRAIN_SETUP_INFO*) sim_setup_info);
		else
			m_sim_instance.sim_control = new SimulationPredictStatusControl(*this, (const _SIM_PREDICT_SETUP_INFO*) sim_setup_info);
		m_sim_instance.sim_control->InitStatusListCtrl(m_ctrStatusGrid);

		m_sim_instance.sim_control_type = _sim_control_type::run;

		ReadySimulation(*sim_setup_info);

		DWORD dwThreadId;
		m_hRunThread = CreateThread(NULL, 0, RunningThread, this, CREATE_SUSPENDED, &dwThreadId);
		ResumeThread(m_hRunThread);
	}
}

void SimulationRunningWnd::PauseRunning()
{
	m_ctrStartBtn.SetWindowText(L"Resume");
	WaitForSingleObject(m_hAwakeFromPause, INFINITE);
	ResetEvent(m_hAwakeFromPause);
	m_ctrStartBtn.SetWindowText(L"Pause");
}

void SimulationRunningWnd::OnBnClickedEnd()
{
	m_sim_instance.sim_control_type = _sim_control_type::stop;
	SetEvent(m_hAwakeFromPause);
}

void SimulationRunningWnd::OnBnClickedButtonStopEpoch()
{
	if (m_sim_instance.sim_control_type == _sim_control_type::stop_epoch)
	{
		m_sim_instance.sim_control_type = _sim_control_type::run;
		m_ctrStartBtn.EnableWindow(TRUE);
		m_ctrStopBtn.EnableWindow(TRUE);
		m_ctrStopEpochBtn.SetWindowText(L"Stop Epoch");
	}
	else
	{
		m_sim_instance.sim_control_type = _sim_control_type::stop_epoch;
		m_ctrStartBtn.EnableWindow(FALSE);// Pause 버튼 클릭을 방지하기 위해
		m_ctrStopBtn.EnableWindow(FALSE);
		m_ctrStopEpochBtn.SetWindowText(L"Continue Epoch");
	}
	SetEvent(m_hAwakeFromPause);
}

void SimulationRunningWnd::dataio_job_status(neuro_u32 job_id, const char* status)
{
	std::wstring run_status = L" run status : ";
	run_status.append(util::StringUtil::MultiByteToWide(status));
	m_ctrRunStatusStatic.SetWindowText(run_status.c_str());
}

DWORD WINAPI SimulationRunningWnd::RunningThread(LPVOID pParam)
{
	DEBUG_OUTPUT(_T("start\r\n"));

	SimulationRunningWnd* _this = (SimulationRunningWnd*)pParam;
	if (!_this)
		return 0L;

	_fail_status ret = _fail_status::none;

	if (_this->m_sim_instance.simulator)
	{
		delete _this->m_sim_instance.simulator;
		_this->m_sim_instance.simulator = NULL;
	}

	_this->m_ctrRunStatusStatic.SetWindowText(L" run status : ready data");

	project::NeuroStudioProject& project = _this->m_simDlg.GetProject();
	dp::model::ProviderModelManager& ipd = project.GetNSManager().GetProvider();
	project::SimDefinition& sim_def = project.GetSimManager();

	simulate::Simulator* simulator = _this->m_sim_instance.sim_control->CreateSimulatorInstance(_this->m_init_share_object
		, ipd
		, *_this->m_network);
	if (!simulator)
	{
		ret = _fail_status::create_simulator;

		DEBUG_OUTPUT(_T("failed creating simulator\r\n"));
		goto end;
	}

	if (!simulator->ReadyToRun())
	{
		ret = _fail_status::net_ready;

		delete simulator;

		DEBUG_OUTPUT(_T("failed to network ready\r\n"));
		goto end;
	}

	_this->m_sim_instance.simulator = simulator;

	_this->m_layerDisplayWnd.SetSimulatorStatus(true);

	_this->m_ctrRunStatusStatic.SetWindowText(L" run status : running");
	if (!_this->m_sim_instance.simulator->Run())
	{
		ret = _fail_status::net_run;
	}

	_this->m_layerDisplayWnd.SetSimulatorStatus(false);

	delete _this->m_sim_instance.simulator;
	_this->m_sim_instance.simulator = NULL;

end:
	if (ret == _fail_status::none)
	{
		_this->m_ctrRunStatusStatic.SetWindowText(L" run status : finished");
		DEBUG_OUTPUT(_T("end\r\n"));
	}
	else
	{
		_this->m_ctrRunStatusStatic.SetWindowText(L" run status : failed");
		DEBUG_OUTPUT(_T("net Train : failed\r\n"));
	}

	_this->m_hRunThread = NULL;
	_this->PostMessage(NPM_COMPLETED_RUN, (WPARAM)ret, NULL);

	return 0;
}

_sigout SimulationRunningWnd::network_signal(const _NET_SIGNAL_INFO& info, std::unordered_set<neuro_u32>* epoch_start_onehot_output)
{
	m_sim_instance.last_elapse = info.total_elapse;

	if (m_sim_instance.sim_control_type == _sim_control_type::finish_condition)
	{
		PatchResults();
		return engine::_sigout::sig_stop;
	}

	_signal_type type = info.GetType();
	if (type == _signal_type::batch_start || type == _signal_type::batch_end)
	{
		BATCH_STATUS_INFO& batch_info = (BATCH_STATUS_INFO&)info;

		auto display_batch_status = [&]()
		{
			{
				thread::Lock::Owner lock(m_sim_instance.m_signal_lock);
				m_sim_instance.batch_signal_vector.push_back(batch_info);
			}
			PostMessage(NPM_SIM_NET_SIGNAL);

			PatchResults();
		};

		engine::_sigout ret;
		if (m_sim_instance.sim_control_type == _sim_control_type::stop)
		{
			display_batch_status();
			ret = engine::_sigout::sig_stop;
		}
		else if (m_sim_instance.sim_control_type == _sim_control_type::pause)
		{
			display_batch_status();

			PauseRunning();

			if (m_sim_instance.sim_control_type == _sim_control_type::stop)
				ret = engine::_sigout::sig_stop;
			else
				ret = engine::_sigout::sig_continue;
		}
		else
		{
			if (m_sim_instance.display_period_batch > 0 && ((batch_info.batch_no + 1) % m_sim_instance.display_period_batch) == 0)
				display_batch_status();

			if (m_sim_instance.sim_control_type == _sim_control_type::stop_epoch)
				ret = engine::_sigout::sig_epoch_stop;
			else
				ret = engine::_sigout::sig_continue;
		}
		return ret;
	}

	PatchResults();

	{
		thread::Lock::Owner lock(m_sim_instance.m_signal_lock);
		m_sim_instance.epoch_signal_vector.push_back(info.Clone());
	}
	PostMessage(NPM_SIM_NET_SIGNAL);

	if (epoch_start_onehot_output)
		*epoch_start_onehot_output = m_onehot_result_output_set;

	return m_sim_instance.sim_control_type == _sim_control_type::stop_epoch ? engine::_sigout::sig_epoch_stop : engine::_sigout::sig_continue;
}

LRESULT SimulationRunningWnd::OnSimNetSignal(WPARAM wParam, LPARAM lParam)
{
	std::wstring elapse_str;
	SimulationStatusControl::GetElapseString(m_sim_instance.last_elapse, elapse_str);
	m_ctrElapseStatic.SetWindowText(elapse_str.c_str());

	thread::Lock::Owner lock(m_sim_instance.m_signal_lock);

	if (m_sim_instance.sim_control == NULL)
		return 0L;

	// 보통은 한개만 있지만, 최악의 경우 마지막의 batch start와 batch end만 한다.
	for(neuro_u32 i= max(0, neuro_32(m_sim_instance.batch_signal_vector.size())-2); i<m_sim_instance.batch_signal_vector.size();i++)
		m_sim_instance.sim_control->NetworkBatchSignalProcess(m_sim_instance.batch_signal_vector[i]);
	m_sim_instance.batch_signal_vector.clear();

	// 보통은 한개만 있지만, 최악의 경우 4개가 있다고 가정하고(start, epoch start, epoch end, end) 
	for (neuro_u32 i = max(0, neuro_32(m_sim_instance.epoch_signal_vector.size()) - 4); i < m_sim_instance.epoch_signal_vector.size(); i++)
	{
		_NET_SIGNAL_INFO* info = m_sim_instance.epoch_signal_vector[i];

		engine::_sigout sigout = m_sim_instance.sim_control->NetworkSignalProcess(*info);
		if (sigout != engine::_sigout::sig_continue)
			m_sim_instance.sim_control_type = _sim_control_type::finish_condition;

		delete info;
	}
	m_sim_instance.epoch_signal_vector.clear();
	return 0L;
}

void SimulationRunningWnd::PostSimulationMessage(const wchar_t* str)
{
	std::wstring* msg = new std::wstring(str);
	PostMessage(NPM_SIM_MESSAGE, (WPARAM)0, (LPARAM)msg);
}

LRESULT SimulationRunningWnd::OnSimMessage(WPARAM wParam, LPARAM lParam)
{
	std::wstring* msg = (std::wstring*) lParam;

	MessageBox(msg->c_str(), L"Learning");

	delete msg;
	return 0L;
}

LRESULT SimulationRunningWnd::OnCompletedRunMessage(WPARAM wParam, LPARAM lParam)
{
	OnSimNetSignal(NULL, NULL);	// 마무리 작업 해야지!

	EndSimulation();

	_fail_status ret = (_fail_status)wParam;
	if (ret == _fail_status::none)
	{
		// 나중에 run이 끝났을때 다르게 표현하자!
		//		if (lParam>=1)
		//			MessageBox(L"Finished", L"Simuation");
	}
	else
	{
		const wchar_t* msg;
		switch (ret)
		{
		case _fail_status::create_simulator:
			msg = L"Failed to create Simulator.\n\nPlease check data providers and Input(or Target) data";
			break;
		case _fail_status::net_ready:
			msg = L"Failed to ready network. It might be from memory lack\n\nPlease set batch size to smaller";
			break;
		case _fail_status::net_run:
			msg = L"Failed to run network. Please ask AI & Human team(ainhuman@gmail.com)";
			break;
		default:
			msg = L"Failed.";
		}
		MessageBox(msg, L"Simuation");
	}

	return 0L;
}

LRESULT SimulationRunningWnd::OnRedrawResults(WPARAM wParam, LPARAM lParam)
{
	m_layerDisplayWnd.RedrawResults();

	return 0L;
}
