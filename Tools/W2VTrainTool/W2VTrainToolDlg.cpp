
// W2VTrainToolDlg.cpp : ���� ����
//

#include "stdafx.h"
#include "W2VTrainTool.h"
#include "W2VTrainToolDlg.h"
#include "afxdialogex.h"
#include "util/StringUtil.h"
#include "util/np_util.h"
#include "util/FileUtil.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// ���� ���α׷� ������ ���Ǵ� CAboutDlg ��ȭ �����Դϴ�.

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// ��ȭ ���� �������Դϴ�.
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV �����Դϴ�.

// �����Դϴ�.
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CW2VTrainToolDlg ��ȭ ����



CW2VTrainToolDlg::CW2VTrainToolDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CW2VTrainToolDlg::IDD, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
	m_load_matrix = NULL;

	m_in_creatdoc = false;
}

CW2VTrainToolDlg::~CW2VTrainToolDlg()
{
	delete m_load_matrix;
}

void CW2VTrainToolDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CW2VTrainToolDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON_CREATE_EXTRACT_TEXT, &CW2VTrainToolDlg::OnBnClickedButtonCreateExtractText)
	ON_BN_CLICKED(IDC_BUTTON_TEXT_INPUT_PATH, &CW2VTrainToolDlg::OnBnClickedButtonTextInputPath)
	ON_BN_CLICKED(IDC_BUTTON_TRAIN_MODEL_PATH, &CW2VTrainToolDlg::OnBnClickedButtonTrainModelPath)
	ON_BN_CLICKED(IDC_BUTTON_TRAIN_TEXT_FILE_PATH, &CW2VTrainToolDlg::OnBnClickedButtonTrainTextFilePath)
	ON_BN_CLICKED(IDC_BUTTON_W2V_TRAIN, &CW2VTrainToolDlg::OnBnClickedButtonW2vTrain)
	ON_BN_CLICKED(IDC_BUTTON_EXECUTE, &CW2VTrainToolDlg::OnBnClickedButtonExecute)
	ON_BN_CLICKED(IDC_BUTTON_LOAD, &CW2VTrainToolDlg::OnBnClickedButtonLoad)
	ON_BN_CLICKED(IDC_CHECK_FASTTEXT_DATA, &CW2VTrainToolDlg::OnBnClickedCheckFasttextData)
	ON_BN_CLICKED(IDC_BUTTON_TEXT_TARGET_PATH, &CW2VTrainToolDlg::OnBnClickedButtonTextTargetPath)
END_MESSAGE_MAP()


void CW2VTrainToolDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// ��ȭ ���ڿ� �ּ�ȭ ���߸� �߰��� ��� �������� �׸�����
//  �Ʒ� �ڵ尡 �ʿ��մϴ�.  ����/�� ���� ����ϴ� MFC ���� ���α׷��� ��쿡��
//  �����ӿ�ũ���� �� �۾��� �ڵ����� �����մϴ�.

void CW2VTrainToolDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // �׸��⸦ ���� ����̽� ���ؽ�Ʈ�Դϴ�.

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Ŭ���̾�Ʈ �簢������ �������� ����� ����ϴ�.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// �������� �׸��ϴ�.
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// ����ڰ� �ּ�ȭ�� â�� ���� ���ȿ� Ŀ���� ǥ�õǵ��� �ý��ۿ���
//  �� �Լ��� ȣ���մϴ�.
HCURSOR CW2VTrainToolDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


BOOL CW2VTrainToolDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// IDM_ABOUTBOX�� �ý��� ��� ������ �־�� �մϴ�.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// �� ��ȭ ������ �������� �����մϴ�.  ���� ���α׷��� �� â�� ��ȭ ���ڰ� �ƴ� ��쿡��
	//  �����ӿ�ũ�� �� �۾��� �ڵ����� �����մϴ�.
	SetIcon(m_hIcon, TRUE);			// ū �������� �����մϴ�.
	SetIcon(m_hIcon, FALSE);		// ���� �������� �����մϴ�.

	((CButton*)GetDlgItem(IDC_CHECK_HAS_HEADER))->SetCheck(BST_CHECKED);

	((CButton*)GetDlgItem(IDC_CHECK_SHUFFLE))->SetCheck(BST_CHECKED);

	SetDlgItemInt(IDC_EDIT_MAX_WORDS_IN_TEXT, 1500);
	SetDlgItemInt(IDC_EDIT_MAX_SENTENCES_IN_TEXT, 0);
	SetDlgItemInt(IDC_EDIT_MAX_WORDS_IN_SENTENCE, 0);

	SetDlgItemInt(IDC_EDIT_FLUSH_COUNT, 1000);

	((CButton*)GetDlgItem(IDC_RADIO_PRINT_VECTORS))->SetCheck(BST_CHECKED);

	SetDlgItemInt(IDC_EDIT_DIM, 50);
	SetDlgItemInt(IDC_EDIT_WS, 5);
	SetDlgItemInt(IDC_EDIT_MIN_COUNT, 5);
	SetDlgItemInt(IDC_EDIT_EPOCH, 20);

	OnBnClickedCheckFasttextData();
	return TRUE;  // ��Ŀ���� ��Ʈ�ѿ� �������� ������ TRUE�� ��ȯ�մϴ�.
}

void CW2VTrainToolDlg::OnBnClickedButtonTextInputPath()
{
	CString old_path;
	GetDlgItemText(IDC_EDIT_TEXT_INPUT_PATH, old_path);

	wchar_t BASED_CODE szFilter[] = L"CSV (*.csv)|*.csv|All Files (*.*)|*.*||";

	CFileDialog	dlg(TRUE, L"*.csv", old_path, OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, szFilter);
	dlg.m_ofn.lpstrTitle = L"CSV file to transform";
	if (dlg.DoModal() != IDOK)
		return;

	SetDlgItemText(IDC_EDIT_TEXT_INPUT_PATH, dlg.GetPathName());
	SetDlgItemText(IDC_EDIT_TEXT_TARGET_PATH, dlg.GetPathName());
}

void CW2VTrainToolDlg::OnBnClickedButtonTextTargetPath()
{
	CString old_path;
	GetDlgItemText(IDC_EDIT_TEXT_TARGET_PATH, old_path);

	wchar_t BASED_CODE szFilter[] = L"text (*.csv;*.txt)|*.csv;*.txt|All Files (*.*)|*.*||";

	CFileDialog	dlg(TRUE, L"*.csv;*.txt", old_path, OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, szFilter);
	dlg.m_ofn.lpstrTitle = L"target file";
	if (dlg.DoModal() != IDOK)
		return;

	SetDlgItemText(IDC_EDIT_TEXT_TARGET_PATH, dlg.GetPathName());
}

#include <thread>

void CW2VTrainToolDlg::OnBnClickedButtonCreateExtractText()
{
	if (m_in_creatdoc)
	{
		MessageBox(L"Currently executing...");
		return;
	}

	CString str;
	GetDlgItemText(IDC_EDIT_TEXT_INPUT_PATH, str);
	if (str.IsEmpty())
	{
		MessageBox(L"no input file");
		return;
	}
	std::string input_path = util::StringUtil::WideToMultiByte((const wchar_t*)str);

	bool isTransformToFastText = ((CButton*)GetDlgItem(IDC_CHECK_FASTTEXT_DATA))->GetCheck() == BST_CHECKED;

	GetDlgItemText(IDC_EDIT_TEXT_TARGET_PATH, str);
	if (str.IsEmpty())
	{
		MessageBox(L"no target file");
		return;
	}
	std::string output_path = util::StringUtil::WideToMultiByte((const wchar_t*)str);
	if (output_path.empty())
	{
		MessageBox(L"no path");
		return;
	}

	bool skip_firstline = ((CButton*)GetDlgItem(IDC_CHECK_SKIP_FIRSTLINE))->GetCheck() == BST_CHECKED;
	bool hasHeader = ((CButton*)GetDlgItem(IDC_CHECK_HAS_HEADER))->GetCheck() == BST_CHECKED;
	if (isTransformToFastText)	// ���� fastText ���̸� header �� ������ ��ŵ�ϰ� �ƴϸ� ��ŵ���� �ʴ´�.
		skip_firstline = hasHeader;

	int setup_max_words = GetDlgItemInt(IDC_EDIT_MAX_WORDS_IN_TEXT);
	int setup_max_sentences = GetDlgItemInt(IDC_EDIT_MAX_SENTENCES_IN_TEXT);
	int setup_max_words_per_sentence = GetDlgItemInt(IDC_EDIT_MAX_WORDS_IN_SENTENCE);
	int axis_to_split = GetDlgItemInt(IDC_EDIT_TRAIN_DATA_AXIS);
	bool shuffle = ((CButton*)GetDlgItem(IDC_CHECK_SHUFFLE))->GetCheck() == BST_CHECKED;

	SetDlgItemText(IDC_EDIT_TOTAL_CONTENTS, L"");
	SetDlgItemText(IDC_EDIT_TOTAL_PARAGRAPHS, L"");
	SetDlgItemText(IDC_EDIT_TOTAL_SENTENCES, L"");
	SetDlgItemText(IDC_EDIT_ELAPSE, L"");

	int flush_count=GetDlgItemInt(IDC_EDIT_FLUSH_COUNT);

	m_in_creatdoc = true;

	np::nlp::CreateW2VTrainDoc create;
	if (create.Create(input_path.c_str(), hasHeader, skip_firstline
		, isTransformToFastText
		, axis_to_split, shuffle
		, setup_max_words, setup_max_sentences, setup_max_words_per_sentence
		, output_path, flush_count, this))
		MessageBox(L"Completed");
	else
		MessageBox(L"Failed");

	m_in_creatdoc = false;
}

void CW2VTrainToolDlg::signal(const np::nlp::_recv_status& status)
{
	SetDlgItemText(IDC_EDIT_TOTAL_CONTENTS, np::util::StringUtil::Transform<wchar_t>(status.total_content).c_str());
//	SetDlgItemText(IDC_EDIT_TOTAL_PARAGRAPHS, np::util::StringUtil::Transform<wchar_t>(status.total_paragraph).c_str());
	SetDlgItemText(IDC_EDIT_TOTAL_SENTENCES, np::util::StringUtil::Transform<wchar_t>(status.total_sentence).c_str());
	SetDlgItemText(IDC_EDIT_ELAPSE, np::util::StringUtil::Transform<wchar_t>(status.elapse).c_str());

	MSG		_msg_;
	while (::PeekMessage(&_msg_, NULL, 0, 0, PM_REMOVE))
	{
		::TranslateMessage(&_msg_);
		::DispatchMessage(&_msg_);
	}
}

void CW2VTrainToolDlg::OnBnClickedButtonTrainTextFilePath()
{
	wchar_t BASED_CODE szFilter[] = L"All Files (*.*)|*.*||";

	CFileDialog	dlg(TRUE, L"*.*", L"", OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, szFilter);
	dlg.m_ofn.lpstrTitle = L"text file to train word2vec";
	if (dlg.DoModal() != IDOK)
		return;

	SetDlgItemText(IDC_EDIT_TRAIN_TEXT_FILE_PATH, dlg.GetPathName());

	CString module_path;
	GetDlgItemText(IDC_EDIT_MODEL_PATH, module_path);
	if (module_path.IsEmpty())
		SetTrainModulePath(dlg.GetPathName());
}

void CW2VTrainToolDlg::OnBnClickedButtonTrainModelPath()
{
	CString old_path;
	GetDlgItemText(IDC_EDIT_MODEL_PATH, old_path);

	wchar_t BASED_CODE szFilter[] = L"All Files (*.*)|*.*||";

	CFileDialog	dlg(FALSE, L"*.*", old_path, OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, szFilter);
	dlg.m_ofn.lpstrTitle = L"W2V model file";
	if (dlg.DoModal() != IDOK)
		return;

	SetTrainModulePath(dlg.GetPathName());
}

void CW2VTrainToolDlg::SetTrainModulePath(const wchar_t* path)
{
	CString train_module_path = util::FileUtil::GetNameFromFileName<wchar_t>(path).c_str();
	SetDlgItemText(IDC_EDIT_MODEL_PATH, train_module_path);
}

#include "util/StringUtil.h"
#include "util/np_util.h"
using namespace np::util;

#include <iostream>

#include "args.h"
void CW2VTrainToolDlg::OnBnClickedButtonW2vTrain()
{
	CString wTrainFilePath;
	GetDlgItemText(IDC_EDIT_TRAIN_TEXT_FILE_PATH, wTrainFilePath);

	CString wModelFilePath;
	GetDlgItemText(IDC_EDIT_MODEL_PATH, wModelFilePath);

	CStringA strTrainPath(wTrainFilePath);
	CStringA strModelFileName(util::FileUtil::GetNameFromFileName<wchar_t>((const wchar_t*)wModelFilePath).c_str());

	if (wTrainFilePath.IsEmpty() || wModelFilePath.IsEmpty())
	{
		MessageBox(L"no path");
		return;
	}

	int dim = GetDlgItemInt(IDC_EDIT_DIM);
	if (dim == 0)
		dim = 100;
	int ws = GetDlgItemInt(IDC_EDIT_WS);
	if (ws < 3)
		ws = 5;
	int min_count = GetDlgItemInt(IDC_EDIT_MIN_COUNT);
	if (min_count < 5)
		min_count = 5;

	int epoch = GetDlgItemInt(IDC_EDIT_EPOCH);
	if (epoch < 1)
		epoch = 5;

	CStringA strDim(StringUtil::Transform<wchar_t>(dim).c_str());
	CStringA strWs(StringUtil::Transform<wchar_t>(ws).c_str());
	CStringA strMinCount(StringUtil::Transform<wchar_t>(min_count).c_str());
	CStringA strEpoch(StringUtil::Transform<wchar_t>(epoch).c_str());

	strModelFileName.Append("_");
	strModelFileName.Append(strDim);
	strModelFileName.Append("dim");

	std::vector<std::string> args({"fastText", "skipgram"
		, "-input", (const char*) strTrainPath
		, "-output", (const char*)strModelFileName
		, "-epoch", (const char*)strEpoch
		, "-minCount", (const char*)strMinCount
		, "-dim", (const char*)strDim
		, "-ws", (const char*)strWs });

	std::shared_ptr<fasttext::Args> a = std::make_shared<fasttext::Args>();
	a->parseArgs(args);

	AllocConsole();
	freopen("CONIN$", "r", stdin);
	freopen("CONOUT$", "w", stdout);
	freopen("CONOUT$", "w", stderr);

	std::cerr << "start train" << std::endl;

	fasttext::FastText fasttext;
	fasttext.train(a);

	FreeConsole();
}

void CW2VTrainToolDlg::OnBnClickedButtonLoad()
{
	CString strModelFilePath;
	GetDlgItemText(IDC_EDIT_MODEL_PATH, strModelFilePath);
	if (!strModelFilePath.IsEmpty())
		strModelFilePath = util::FileUtil::GetNameFromFileName<wchar_t>((const wchar_t*)strModelFilePath).c_str();

	wchar_t BASED_CODE szFilter[] = L"W2V model (*.bin; *.vec)|*.bin;*.vec|All Files (*.*)|*.*||";

	CFileDialog	dlg(TRUE, L"*.bin;*.vec", strModelFilePath, OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, szFilter);
	dlg.m_ofn.lpstrTitle = L"W2V modefl file";
	if (dlg.DoModal() != IDOK)
		return;

	strModelFilePath = dlg.GetPathName();
	SetDlgItemText(IDC_EDIT_TEST_MODEL_PATH, strModelFilePath);
	if (strModelFilePath.IsEmpty())
	{
		MessageBox(L"no model file");
		return;
	}

	WCHAR szExt[_MAX_EXT];
	_wsplitpath(strModelFilePath, NULL, NULL, NULL, szExt);

	std::string strPath((const char*)CStringA(strModelFilePath));

	np::Timer timer;
	if (wcscmp(szExt, L".bin")==0)
		m_fasttext.loadModel(strPath);
	else
		m_fasttext.loadVectors(strPath, true);

	if (m_load_matrix)
		delete m_load_matrix;

	m_load_matrix = new fasttext::Matrix(m_fasttext.getDictionary()->nwords(), m_fasttext.getDimension());
	m_fasttext.precomputeWordVectors(*m_load_matrix);

	MessageBox(util::StringUtil::Format<wchar_t>(L"completed load. %f elapse", timer.elapsed()).c_str());
}

#include <queue>

void CW2VTrainToolDlg::OnBnClickedButtonExecute()
{
	if (m_fasttext.getDimension() == 0 || !m_load_matrix)
	{
		MessageBox(L"load w2v model file");
		return;
	}

	CString strQuery;
	GetDlgItemText(IDC_EDIT_QUERY, strQuery);
	if (strQuery.IsEmpty())
	{
		MessageBox(L"input query word");
		return;
	}

	bool print_vector = ((CButton*)GetDlgItem(IDC_RADIO_PRINT_VECTORS))->GetCheck() == BST_CHECKED;

	CListCtrl* ctrList = (CListCtrl*)GetDlgItem(IDC_LIST_RESULT);
	ctrList->DeleteAllItems();
	CHeaderCtrl* pHeaderCtrl = ctrList->GetHeaderCtrl();
	int nColumn = pHeaderCtrl->GetItemCount();
	for (int i = 0; i < nColumn;i++)
		ctrList->DeleteColumn(0);

	fasttext::Vector queryVec(m_fasttext.getDimension());

	std::string query_word = util::StringUtil::WideToMultiByte((const wchar_t*)strQuery);

	if (print_vector)
	{
		ctrList->InsertColumn(0, L"value");

		m_fasttext.getVector(queryVec, query_word);

		for (int64_t j = 0; j < queryVec.m_; j++)
			ctrList->InsertItem(j, StringUtil::Transform<wchar_t>(queryVec.data_[j]).c_str());
	}
	else
	{
		ctrList->InsertColumn(0, L"word");
		ctrList->InsertColumn(1, L"value");

		int k = GetDlgItemInt(IDC_EDIT_NEAR_K);

		std::set<std::string> banSet;
		banSet.insert(query_word);
		m_fasttext.getVector(queryVec, query_word);
		{
			fasttext::real queryNorm = queryVec.norm();
			if (std::abs(queryNorm) < 1e-8) {
				queryNorm = 1;
			}
			std::priority_queue<std::pair<fasttext::real, std::string>> heap;

			const std::shared_ptr<const fasttext::Dictionary>& dic = m_fasttext.getDictionary();
			for (int32_t i = 0, n = dic->nwords(); i < n; i++) 
			{
				std::string word = dic->getWord(i);
				fasttext::real dp = m_load_matrix->dotRow(queryVec, i);
				heap.push(std::make_pair(dp / queryNorm, word));
			}
			int32_t i = 0;
			while (i < k && heap.size() > 0) 
			{
				auto it = banSet.find(heap.top().second);
				if (it == banSet.end())
				{
					ctrList->InsertItem(i, CString(heap.top().second.c_str()));
					ctrList->SetItemText(i, 1, StringUtil::Transform<wchar_t>(heap.top().first).c_str());
					i++;
				}
				heap.pop();
			}	
		}
		m_fasttext.findNN(*m_load_matrix, queryVec, k, banSet);
	}
	
	nColumn = pHeaderCtrl->GetItemCount();
	for (int i = 0; i<nColumn; i++)
		ctrList->SetColumnWidth(i, LVSCW_AUTOSIZE_USEHEADER);
}

void CW2VTrainToolDlg::OnBnClickedCheckFasttextData()
{
	bool isTransformToFastText = ((CButton*)GetDlgItem(IDC_CHECK_FASTTEXT_DATA))->GetCheck()==BST_CHECKED;
	GetDlgItem(IDC_EDIT_TRAIN_DATA_AXIS)->EnableWindow(!isTransformToFastText);
	GetDlgItem(IDC_CHECK_SHUFFLE)->EnableWindow(!isTransformToFastText);

	GetDlgItem(IDC_EDIT_MAX_WORDS_IN_TEXT)->EnableWindow(!isTransformToFastText);
	GetDlgItem(IDC_EDIT_MAX_SENTENCES_IN_TEXT)->EnableWindow(!isTransformToFastText);
	GetDlgItem(IDC_EDIT_MAX_WORDS_IN_SENTENCE)->EnableWindow(!isTransformToFastText);

	if (isTransformToFastText)
	{
		bool hasHeader = ((CButton*)GetDlgItem(IDC_CHECK_HAS_HEADER))->GetCheck() == BST_CHECKED;
		((CButton*)GetDlgItem(IDC_CHECK_SKIP_FIRSTLINE))->SetCheck(hasHeader ? BST_CHECKED : BST_UNCHECKED);
	}
}
