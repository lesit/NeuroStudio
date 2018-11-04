#pragma once

#include "NeuroData/model/DataProviderModel.h"
#include "NeuroData/reader/DataSourceNameVector.h"
#include "storage/DeviceAdaptor.h"
#include "project/SimDefinition.h"

using namespace np::dp::model;
using namespace np::dp::preprocessor;

class CSimDataTreeCtrl : public CTreeCtrl
{
public:
	CSimDataTreeCtrl();
	virtual ~CSimDataTreeCtrl();

	void InitProvider(const dp::model::_producer_model_vector& producer_vector, const dp::preprocessor::_uid_datanames_map& uid_str_vector_map);
	void GetDataFiles(_uid_datanames_map& uid_datanames_map) const;

protected:
	friend class CSimDataTreeWnd;

	enum class _tree_item_type { none, producer, file };
	struct _TREE_DATA
	{
		virtual ~_TREE_DATA() {}

		virtual _tree_item_type GetItemType() const = 0;
	};

	struct _PRODUCER_TREE_DATA : public _TREE_DATA
	{
		_PRODUCER_TREE_DATA(const dp::model::AbstractProducerModel* _producer_model)
		{
			producer_model = _producer_model;
		}
		virtual ~_PRODUCER_TREE_DATA() {}

		virtual _tree_item_type GetItemType() const { return _tree_item_type::producer; }

		const dp::model::AbstractProducerModel* producer_model;

		DataSourceBasePath<char> basedir;
	};

	struct _FILE_TREE_DATA : public _TREE_DATA
	{
		_FILE_TREE_DATA(neuro_u32 basedir_id = neuro_last32)
		{
			this->basedir_id = basedir_id;
		}
		virtual ~_FILE_TREE_DATA() {}

		virtual _tree_item_type GetItemType() const { return _tree_item_type::file; }

		neuro_u32 basedir_id;
	};

	void GetDataFiles(HTREEITEM hProducer, const _PRODUCER_TREE_DATA& tree_item, DataSourceNameVector<char>& path_vector) const;
	void ClearProdocuerFiles(HTREEITEM hProducer);

	_tree_item_type GetItemType(HTREEITEM hItem) const;

	_PRODUCER_TREE_DATA* GetProducerTreeData(HTREEITEM hItem, HTREEITEM& hProducer) const;

	void AddDataFiles(HTREEITEM hProducer, const DataSourceNameVector<char>& path_vector, bool bClearPrevious);
	void AddDirPath(HTREEITEM hProducer, const std::string& path);
	void AddFilePath(HTREEITEM hProducer, _PRODUCER_TREE_DATA& tree_data, const std::string& path);
	void DeleteSelectedItem();
	void MoveSelectedItem(bool bMoveUp);

protected:
	DECLARE_MESSAGE_MAP()
	afx_msg void OnTvnDeleteItem(NMHDR *pNMHDR, LRESULT *pResult);
};

class CSimDataTreeWnd : public CWnd
{
public:
	CSimDataTreeWnd();
	virtual ~CSimDataTreeWnd();
	
	CSimDataTreeCtrl m_ctrTree;

protected:
	CEdit m_ctrFullpathEdit;

	CButton m_ctrAddBtn;
	CButton m_ctrDelBtn;
	CButton m_ctrUpBtn;
	CButton m_ctrDownBtn;
	CButton m_ctrCreateSubdirBtn;

protected:
	DECLARE_MESSAGE_MAP()
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnTvnSelchanged(NMHDR *pNMHDR, LRESULT *pResult);
	afx_msg void OnBnClickedAddData();
	afx_msg void OnBnClickedDelData();
	afx_msg void OnBnClickedUpData();
	afx_msg void OnBnClickedDownData();
	afx_msg void OnBnClickedCreateSubdirs();
};
