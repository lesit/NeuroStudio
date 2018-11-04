#include "stdafx.h"

#include "SaveNSProject.h"
#include <xercesc/dom/DOMDocument.hpp>

#include "NSProjectXML.h"

#include "util/FileUtil.h"

using namespace np;
using namespace np::project;
using namespace np::project::xml;
//using namespace np::simulate;

SaveNSProject::SaveNSProject(NeuroStudioProject& project)
: m_project(project)
, m_provider(project.GetNSManager().GetProvider())
{
	m_document=NULL;
}

bool SaveNSProject::Save(const char* strFilePath)
{
	NSProjectXML xml;
	m_document=xml.CreateDocument();
	if(!m_document)
		return false;

	XERCES_CPP_NAMESPACE::DOMElement* rootElem = m_document->getDocumentElement();
	if(!rootElem)
		return false;

	m_proj_dir = util::FileUtil::GetDirFromPath<char>(strFilePath);
#ifdef _WINDOWS
	m_proj_dir += "\\";
#else
	m_proj_dir += "/";
#endif

	{	// network 저장
		XERCES_CPP_NAMESPACE::DOMElement* elem = CreateNeuralNetworkElem();
		if (!elem)
			return false;

		rootElem->appendChild(elem);
	}

	{	// data preprocessor 저장
		XERCES_CPP_NAMESPACE::DOMElement* elem = CreateDataPreprocessorElem();
		if (!elem)
			return false;

		rootElem->appendChild(elem);
	}

	{	// stream list들 저장
		XERCES_CPP_NAMESPACE::DOMElement* elem = CreateSimulationElem();
		if (!elem)
			return false;

		rootElem->appendChild(elem);
	}

	m_document=NULL;
	return xml.SaveXML(util::StringUtil::MultiByteToWide(strFilePath).c_str());
}

XERCES_CPP_NAMESPACE::DOMElement* SaveNSProject::CreateNeuralNetworkElem()
{
	if (!m_project.HasDeviceFactory())
		return NULL;

	XERCES_CPP_NAMESPACE::DOMElement* elem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szNeuralNetwork);

	const char* file_path = m_project.GetNetworkFilePath();
	if (file_path)
	{
		XERCES_CPP_NAMESPACE::DOMElement* nn_deviceElem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szFileDevice);

		std::wstring file_name = util::StringUtil::MultiByteToWide(util::FileUtil::GetFileName(file_path));

		nn_deviceElem->setAttribute(att::szFileName, file_name.c_str());

		elem->appendChild(nn_deviceElem);
	}
	return elem;
}

XERCES_CPP_NAMESPACE::DOMElement* SaveNSProject::CreateDataPreprocessorElem()
{
	XERCES_CPP_NAMESPACE::DOMElement* elem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szDataPreprocessor);

	if (m_provider.GetLearnProvider())
	{
		XERCES_CPP_NAMESPACE::DOMElement* childElem = CreateProviderElem(elem::szLearnProvider, *m_provider.GetLearnProvider());
		elem->appendChild(childElem);

		childElem = CreateProviderElem(elem::szPredictProvider, m_provider.GetPredictProvider());
		elem->appendChild(childElem);
	}
	else
	{
		XERCES_CPP_NAMESPACE::DOMElement* childElem = CreateProviderElem(elem::szIntegratedProvider, m_provider.GetPredictProvider());
		elem->appendChild(childElem);
	}

	return elem;
}

XERCES_CPP_NAMESPACE::DOMElement* SaveNSProject::CreateProviderElem(const wchar_t* provider_name, const model::DataProviderModel& provider)
{
	XERCES_CPP_NAMESPACE::DOMElement* elem = m_document->createElementNS(namespace_uri::szNamespaceUri, provider_name);

	{
		XERCES_CPP_NAMESPACE::DOMElement* listElem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szDataReaderList);

		const model::_reader_model_vector& model_vector = provider.GetReaderVector();
		for (neuro_u32 i = 0; i<model_vector.size(); i++)
		{
			AbstractReaderModel* model = model_vector[i];
			XERCES_CPP_NAMESPACE::DOMElement* child = CreateReaderElem(*model);
			if (!child)
				continue;

			listElem->appendChild(child);
		}
		elem->appendChild(listElem);
	}
	{
		XERCES_CPP_NAMESPACE::DOMElement* listElem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szProducerList);

		const _producer_model_vector& model_vector = provider.GetProducerVector();
		for (neuro_u32 i = 0; i<model_vector.size(); i++)
		{
			AbstractProducerModel* model = model_vector[i];
			XERCES_CPP_NAMESPACE::DOMElement* child = CreateProducerElem(*model);
			if (!child)
				continue;

			listElem->appendChild(child);
		}
		elem->appendChild(listElem);
	}
	return elem;
}

XERCES_CPP_NAMESPACE::DOMElement* SaveNSProject::CreateReaderElem(const AbstractReaderModel& model)
{
	XERCES_CPP_NAMESPACE::DOMElement* elem=NULL;
	if (model.GetReaderType() == _reader_type::text)
	{
		elem=m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szTextReader);

		const TextReaderModel& text_reader = (const TextReaderModel&)model;

		elem->setAttribute(att::szStartSkip, util::StringUtil::Transform<wchar_t>(text_reader.GetSkipFirstCount()).c_str());
		elem->setAttribute(att::szReverse, text_reader.IsReverse() ? att_value::boolean::szTrue : att_value::boolean::szFalse);

		elem->setAttribute(att::szImported, util::StringUtil::MultiByteToWide(text_reader.GetImportedSource()).c_str());

		const dp::TextColumnDelimiter& delimiter = text_reader.GetDelimiter();
		if (delimiter.GetType() == dp::_delimiter_type::token)
		{
			elem->setAttribute(att::szDelimiterType, att_value::delimiterType::szToken);

			const ExtTextColumnTokenDelimiter& token_delimiter = (const ExtTextColumnTokenDelimiter&)delimiter;

			elem->setAttribute(att::szColumnCount, util::StringUtil::Transform<wchar_t>(token_delimiter.m_column_count).c_str());

			elem->setAttribute(att::szContentToken, util::StringUtil::MultiByteToWide(util::StringUtil::TransformAsciiToText(token_delimiter.content_token)).c_str());
			elem->setAttribute(att::szDoubleQuotes, token_delimiter.double_quote ? att_value::boolean::szTrue : att_value::boolean::szFalse);
			elem->setAttribute(att::szColumnToken, util::StringUtil::MultiByteToWide(TokenVectorToString(token_delimiter.column_token_vector)).c_str());
		}
		else if (delimiter.GetType() == dp::_delimiter_type::length)
		{
			elem->setAttribute(att::szDelimiterType, att_value::delimiterType::szLength);

			const dp::TextColumnLengthDelimiter& lengh_delimiter = (const dp::TextColumnLengthDelimiter&)delimiter;
			for (neuro_u32 i = 0; i < lengh_delimiter.lengh_vector.size(); i++)
			{
				XERCES_CPP_NAMESPACE::DOMElement* itemElem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szItem);
				itemElem->setAttribute(att::szColumnLength, util::StringUtil::Transform<wchar_t>(lengh_delimiter.lengh_vector[i]).c_str());
				elem->appendChild(itemElem);
			}
		}
	}
	else if (model.GetReaderType() == _reader_type::text)
	{
		elem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szBinaryReader);

		const BinaryReaderModel& bin_reader = (const BinaryReaderModel&)model;

		const _data_type_vector& type_vector = bin_reader.GetTypeVector();
		for (neuro_u32 i = 0; i < type_vector.size(); i++)
		{
			XERCES_CPP_NAMESPACE::DOMElement* itemElem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szItem);
			itemElem->setAttribute(att::szColumnDataType, _data_type_string[(int)type_vector[i]]);
			elem->appendChild(itemElem);
		}
	}
	if(!elem)
		return NULL;

	elem->setAttribute(att::szID, util::StringUtil::Transform<wchar_t>(model.uid).c_str());
	if(model.GetInput())
		elem->setAttribute(att::szInputID, util::StringUtil::Transform<wchar_t>(model.GetInput()->uid).c_str());
	return elem;
}

XERCES_CPP_NAMESPACE::DOMElement* SaveNSProject::CreateProducerElem(const AbstractProducerModel& producer_model)
{
	XERCES_CPP_NAMESPACE::DOMElement* elem=NULL;

	model::_producer_type producer_type = producer_model.GetProducerType();

	if (producer_type == model::_producer_type::image_file)
	{
		elem=m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szImageProducer);

		const ImageFileProducerModel& model = (const ImageFileProducerModel&)producer_model;

		const _NEURO_INPUT_IMAGEFILE_INFO& info= model.GetDefinition();

		elem->setAttribute(att::szWidth, util::StringUtil::Transform<wchar_t>(info.sz.width).c_str());
		elem->setAttribute(att::szHeight, util::StringUtil::Transform<wchar_t>(info.sz.height).c_str());

		std::wstring color_type;
		switch(info.color_type)
		{
		case _color_type::mono:
			color_type = att_value::imgScaleType::szMono;
			break;
		case _color_type::rgb:
			color_type = att_value::imgScaleType::szRGB;
			break;
		}
		if (color_type.empty())
		{
			elem->release();
			return NULL;
		}
		elem->setAttribute(att::szScaleType, color_type.c_str());
	}
	else if (producer_type == model::_producer_type::windll)
	{
		elem=m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szAddinProducer);
	}
	else if (producer_type == model::_producer_type::mnist_img)
	{
		elem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szMnistImgProducer);
	}
	else if (producer_type == model::_producer_type::mnist_label)
	{
		elem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szMnistLabelProducer);
	}
	else if (producer_type == model::_producer_type::numeric)
	{
		const NumericProducerModel& model = (NumericProducerModel&)producer_model;

		elem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szNumericProducer);

		neuro_u32 onehot_size = model.GetLabelOutCount();
		if (onehot_size > 0)
			elem->setAttribute(att::szOnehotSize, util::StringUtil::Transform<wchar_t>(onehot_size).c_str());

		XERCES_CPP_NAMESPACE::DOMElement* listElem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szItemList);
		elem->appendChild(listElem);

		const std::map<neuro_u32, neuro_u32>& using_index_ma_map = model.GetUsingSourceColumns();
		std::map<neuro_u32, neuro_u32>::const_iterator it = using_index_ma_map.begin();
		for(;it!= using_index_ma_map.end();it++)
		{
			XERCES_CPP_NAMESPACE::DOMElement* valueElem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szItem);

			valueElem->setAttribute(att::szIndex, util::StringUtil::Transform<wchar_t>(it->first).c_str());
			valueElem->setAttribute(att::szMA, util::StringUtil::Transform<wchar_t>(it->second).c_str());

			listElem->appendChild(valueElem);
		}
	}
	else if (producer_type == model::_producer_type::nlp)
	{
		NlpProducerModel& model = (NlpProducerModel&)producer_model;

		elem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szNLPProducer);

		{
			wchar_t szFilePath[256];
#if defined(WIN32) | defined(WIN64)
			GetModuleFileName(NULL, szFilePath, _countof(szFilePath));
#else
			memset(szFilePath, 0, sizeof(szFilePath));	// 흠.. 없으면 그냥 실행한 위치니까 상관 없을듯!
#endif
			// 이게 아니라 simulator에서 실제 지정하고, 저장해야 한다!
			std::string base_dir = util::StringUtil::WideToMultiByte(util::FileUtil::GetDirFromPath<wchar_t>(szFilePath));

			XERCES_CPP_NAMESPACE::DOMElement* mecap_rc_elem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szMecab);

			std::string mecab_path = util::FileUtil::GetRelativePath<char>(base_dir, model.GetMecapRcPath()).c_str();
			mecap_rc_elem->setAttribute(att::szFileName, util::StringUtil::MultiByteToWide(mecab_path).c_str());
			elem->appendChild(mecap_rc_elem);

			XERCES_CPP_NAMESPACE::DOMElement* w2v_path_elem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szW2V);
			std::string w2v_path = util::FileUtil::GetRelativePath<char>(base_dir, model.GetWordToVectorPath()).c_str();
			w2v_path_elem->setAttribute(att::szFileName, util::StringUtil::MultiByteToWide(w2v_path).c_str());
			elem->appendChild(w2v_path_elem);
		}

		XERCES_CPP_NAMESPACE::DOMElement* listElem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szItemList);
		elem->appendChild(listElem);

		const _u32_set& index_set = model.GetUsingSourceColumnSet();
		for (_u32_set::const_iterator it = index_set.begin(); it != index_set.end(); it++)
		{
			XERCES_CPP_NAMESPACE::DOMElement* valueElem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szItem);

			valueElem->setAttribute(att::szIndex, util::StringUtil::Transform<wchar_t>(*it).c_str());

			listElem->appendChild(valueElem);
		}

		elem->setAttribute(att::szUseMorphemeParser, model.UseMorphemeParser() ? att_value::boolean::szTrue : att_value::boolean::szFalse);
		elem->setAttribute(att::szUseMorphemeTypeVector, model.UseMorphemeTypeVector() ? att_value::boolean::szTrue : att_value::boolean::szFalse);

		elem->setAttribute(att::szWordVectorNom, model.IsVectorNormalization() ? att_value::boolean::szTrue : att_value::boolean::szFalse);

		elem->setAttribute(att::szParseSentence, model.ParsingSentence() ? att_value::boolean::szTrue : att_value::boolean::szFalse);

		elem->setAttribute(att::szMaxWord, util::StringUtil::Transform<wchar_t>(model.GetMaxWord()).c_str());
		elem->setAttribute(att::szMaxSentence, util::StringUtil::Transform<wchar_t>(model.GetMaxSentence()).c_str());
		elem->setAttribute(att::szMaxWordPerSentence, util::StringUtil::Transform<wchar_t>(model.GetMaxWordPerSentence()).c_str());
	}
	else if (producer_type == model::_producer_type::increase_predict)
	{
		elem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szIncreasePredictProducer);

		IncreasePredictProducerModel& model = (IncreasePredictProducerModel&)producer_model;

		elem->setAttribute(att::szPredictDistance, util::StringUtil::Transform<wchar_t>(model.GetPredictDistance()).c_str());
		elem->setAttribute(att::szIndex, util::StringUtil::Transform<wchar_t>(model.GetSourceColumn()).c_str());
		elem->setAttribute(att::szMA, util::StringUtil::Transform<wchar_t>(model.GetMovingAvarage()).c_str());

		switch (model.GetPredictType())
		{
		case _increase_predict_type::value:
			elem->setAttribute(att::szPreprocessType, att_value::classRangePreprocessType::szIncrease);
			break;
		case _increase_predict_type::rate:
			elem->setAttribute(att::szPreprocessType, att_value::classRangePreprocessType::szIncreaseRate);
			break;
		}

		const _predict_range_vector& ranges = model.GetRanges();
		for (size_t i = 0, n = ranges.size(); i < n; i++)
		{
			XERCES_CPP_NAMESPACE::DOMElement* rangeElem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szRange);
			rangeElem->setAttribute(att::szValue, util::StringUtil::Transform<wchar_t>(ranges[i].value).c_str());

			std::wstring ineuality;
			switch (ranges[i].ineuality)
			{
			case _inequality_type::less:
				ineuality = L"<";
				break;
			case _inequality_type::equal_less:
				ineuality = L"<=";
				break;
			case _inequality_type::equal_greater:
				ineuality = L">=";
				break;
			case _inequality_type::greater:
				ineuality = L">";
				break;
			}
			rangeElem->setAttribute(att::szInequality, ineuality.c_str());

			elem->appendChild(rangeElem);
		}
	}
	else
		return NULL;

	elem->setAttribute(att::szID, util::StringUtil::Transform<wchar_t>(producer_model.uid).c_str());
	if (producer_model.GetInput())
		elem->setAttribute(att::szInputID, util::StringUtil::Transform<wchar_t>(producer_model.GetInput()->uid).c_str());

	elem->setAttribute(att::szStartPosition, util::StringUtil::Transform<wchar_t>(producer_model.GetStartPosition()).c_str());

	if (producer_model.GetBindingSet().size() > 0)
	{
		XERCES_CPP_NAMESPACE::DOMElement* binding_list_elem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szBindingList);
		elem->appendChild(binding_list_elem);

		const np::_neuro_binding_model_set& binding_set = producer_model.GetBindingSet();
		_neuro_binding_model_set::const_iterator it = binding_set.begin();
		for (; it != binding_set.end(); it++)
		{
			XERCES_CPP_NAMESPACE::DOMElement* binding_elem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szBinding);
			binding_list_elem->appendChild(binding_elem);

			binding_elem->setAttribute(att::szID, util::StringUtil::Transform<wchar_t>((*it)->GetUniqueID()).c_str());
		}
	}
	return elem;
}

XERCES_CPP_NAMESPACE::DOMElement* SaveNSProject::CreateSimulationElem()
{
	XERCES_CPP_NAMESPACE::DOMElement* elem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szSimulation);

	{
		XERCES_CPP_NAMESPACE::DOMElement* child = CreateSimResultViewListElem();
		elem->appendChild(child);
	}

	{
		XERCES_CPP_NAMESPACE::DOMElement* child = CreateSimEnvElem();
		elem->appendChild(child);
	}

	{
		XERCES_CPP_NAMESPACE::DOMElement* child = CreateSimTrainDataElem();
		elem->appendChild(child);
	}

	{
		XERCES_CPP_NAMESPACE::DOMElement* child = CreateSimPredictDataElem();
		elem->appendChild(child);
	}

	return elem;
}

XERCES_CPP_NAMESPACE::DOMElement* SaveNSProject::CreateSimResultViewListElem()
{
	XERCES_CPP_NAMESPACE::DOMElement* elem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szResultViewList);

	const project::SimDefinition& sim_def = m_project.GetSimManager();
	const project::_layer_display_info_map& layer_display_type_map = sim_def.GetLayerDisplayInfoMap();
	_layer_display_info_map::const_iterator it = layer_display_type_map.begin();
	for (; it != layer_display_type_map.end(); it++)
	{
		const wchar_t* strType=NULL;
		switch (it->second.type)
		{
		case project::_layer_display_type::image:
			strType = att_value::simViewType::szImage;
			break;
		case project::_layer_display_type::list:
			strType = att_value::simViewType::szList;
			break;
/*		case project::_layer_display_type::graph:
			strType = att_value::simViewType::szGraph;
			break;*/
		default:
			continue;
		}

		XERCES_CPP_NAMESPACE::DOMElement* view_elem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szResultView);
		view_elem->setAttribute(att::szLayerUID, util::StringUtil::Transform<wchar_t>(it->first).c_str());
		view_elem->setAttribute(att::szType, strType);
		if(it->second.is_argmax_output)
			view_elem->setAttribute(att::szArgmaxOutput, att_value::boolean::szTrue);
		if(it->second.is_onehot_analysis_result)
			view_elem->setAttribute(att::szOnehotAnalysisResult, att_value::boolean::szTrue);

		elem->appendChild(view_elem);
	}
	return elem;
}

XERCES_CPP_NAMESPACE::DOMElement* SaveNSProject::CreateSimEnvElem()
{
	XERCES_CPP_NAMESPACE::DOMElement* elem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szEnv);

	SimDefinition& sim = m_project.GetSimManager();
	{
		XERCES_CPP_NAMESPACE::DOMElement* trainElem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szTrainEnv);

		const SIM_TRAIN_ENV& env = sim.GetTrainEnv();

		trainElem->setAttribute(att::szMinibatchSize, util::StringUtil::Transform<wchar_t>(env.minibatch_size).c_str());

		if (!env.useNdf)
			trainElem->setAttribute(att::szUseNdf, att_value::boolean::szFalse);

		if (env.data_noising)
			trainElem->setAttribute(att::szDataNoising, att_value::boolean::szTrue);

		XERCES_CPP_NAMESPACE::DOMElement* child = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szEndCondition);
		child->setAttribute(att::szMaxEpoch, util::StringUtil::Transform<wchar_t>(env.max_epoch).c_str());

		if (env.is_end_below_error)
		{
			XERCES_CPP_NAMESPACE::DOMElement* condElem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szBelowError);
			condElem->setAttribute(att::szError, util::StringUtil::Transform<wchar_t>(env.close_error).c_str());

			child->appendChild(condElem);
		}

		trainElem->appendChild(child);

		child = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szDisplay);
		child->setAttribute(att::szPeriodSample, util::StringUtil::Transform<wchar_t>(env.display_period_sample).c_str());
		trainElem->appendChild(child);

		child = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szAnalyze);
		trainElem->appendChild(child);
		if (env.bAnalyzeArgmaxAccuracy)
			child->setAttribute(att::szArgmaxAccuracy, att_value::boolean::szTrue);
		if (env.bAnalyzeLossHistory)
			child->setAttribute(att::szLossHistory, att_value::boolean::szTrue);
		if (!env.bTestAfterLearn)
			child->setAttribute(att::szTestAfterLearn, att_value::boolean::szFalse);

		elem->appendChild(trainElem);
	}

	{
		XERCES_CPP_NAMESPACE::DOMElement* runElem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szRunEnv);

		const SIM_RUN_ENV& env = sim.GetRunEnv();

		runElem->setAttribute(att::szMinibatchSize, util::StringUtil::Transform<wchar_t>(env.minibatch_size).c_str());

		XERCES_CPP_NAMESPACE::DOMElement* child = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szDisplay);
		child->setAttribute(att::szPeriodSample, util::StringUtil::Transform<wchar_t>(env.display_period_sample).c_str());
		runElem->appendChild(child);

		elem->appendChild(runElem);
	}
	return elem;
}

XERCES_CPP_NAMESPACE::DOMElement* SaveNSProject::CreateSimTrainDataElem()
{
	XERCES_CPP_NAMESPACE::DOMElement* trainDataElem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szTrainData);

	XERCES_CPP_NAMESPACE::DOMElement* trainLearnElem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szTrainLearn);
	CompositeSimDataElem(trainLearnElem, m_project.GetSimManager().GetLastLearnData());
	trainDataElem->appendChild(trainLearnElem);
	
	XERCES_CPP_NAMESPACE::DOMElement* trainTestElem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szTrainTest);
	CompositeSimDataElem(trainTestElem, m_project.GetSimManager().GetLastTestData());
	trainDataElem->appendChild(trainTestElem);
	return trainDataElem;
}

XERCES_CPP_NAMESPACE::DOMElement* SaveNSProject::CreateSimPredictDataElem()
{
	XERCES_CPP_NAMESPACE::DOMElement* predictElem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szPredictData);

	XERCES_CPP_NAMESPACE::DOMElement* InputElem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szPredictInput);
	predictElem->appendChild(InputElem);
	CompositeSimDataElem(InputElem, m_project.GetSimManager().GetLastPredictData());

	XERCES_CPP_NAMESPACE::DOMElement* OutputElem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szPredictOutput);
	predictElem->appendChild(OutputElem);
	std::wstring output_path = m_project.GetSimManager().GetLastPredictOutputPath();
	if (!output_path.empty())
	{
		XERCES_CPP_NAMESPACE::DOMElement* fileElem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szFile);
		fileElem->setAttribute(att::szFilePath, output_path.c_str());
		OutputElem->appendChild(fileElem);
	}
	const dp::_STREAM_WRITE_INFO& write_info=m_project.GetSimManager().GetLastPredictOutputWriteInfo();
	OutputElem->setAttribute(att::szOutputNoPrefix, util::StringUtil::MultiByteToWide(write_info.no_type_prefix).c_str());

	return predictElem;
}

void SaveNSProject::CompositeSimDataElem(XERCES_CPP_NAMESPACE::DOMElement* elem, const dp::preprocessor::_uid_datanames_map& source_map)
{
	dp::preprocessor::_uid_datanames_map::const_iterator it = source_map.begin();
	for (; it != source_map.end(); it++)
	{
		XERCES_CPP_NAMESPACE::DOMElement* itemElem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szItem);
		elem->appendChild(itemElem);

		itemElem->setAttribute(att::szID, util::StringUtil::Transform<wchar_t>(it->first).c_str());

		const DataSourceNameVector<char>& source_name_vector = it->second;
		for (neuro_u32 i = 0, n = source_name_vector.GetCount(); i < n; i++)
		{
			std::string relative_path = util::FileUtil::GetRelativePath<char>(m_proj_dir, source_name_vector.GetPath(i));

			XERCES_CPP_NAMESPACE::DOMElement* fileElem = m_document->createElementNS(namespace_uri::szNamespaceUri, elem::szFile);
			fileElem->setAttribute(att::szFilePath, util::StringUtil::MultiByteToWide(relative_path).c_str());
			itemElem->appendChild(fileElem);
		}
	}
}
