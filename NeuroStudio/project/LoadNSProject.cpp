#include "stdafx.h"

#include "LoadNSProject.h"
#include <xercesc/dom/DOMDocument.hpp>

#include "NSProjectXML.h"

#include "util/FileUtil.h"

using namespace np;
using namespace np::project;
using namespace np::project::xml;

LoadNSProject::LoadNSProject(NeuroStudioProject& project)
: m_project(project)
, m_nsManager(m_project.GetNSManager())
, m_provider(m_nsManager.GetProvider())
{
}

bool LoadNSProject::Load(const char* strFilePath)
{
	std::wstring project_path = util::StringUtil::MultiByteToWide(strFilePath);

	NSProjectXML xml;
	XERCES_CPP_NAMESPACE::DOMDocument* document = xml.LoadXML(project_path.c_str());
	if(!document)
		return false;

	const XERCES_CPP_NAMESPACE::DOMElement* rootElem=document->getDocumentElement();
	if(!rootElem)
		return false;

	m_proj_dir = util::FileUtil::GetDirFromPath<char>(strFilePath).append("\\");

	const XERCES_CPP_NAMESPACE::DOMElement* childElem=rootElem->getFirstElementChild();

	if (wcscmp(childElem->getLocalName(), elem::szNeuralNetwork) == 0)
	{
		LoadNeuralNetwork(childElem);
		childElem = childElem->getNextElementSibling();
	}

	if (childElem && wcscmp(childElem->getLocalName(), elem::szDataPreprocessor) == 0)
	{
		LoadDataPreprocessor(childElem);
		childElem = childElem->getNextElementSibling();
	}

	if (childElem && wcscmp(childElem->getLocalName(), elem::szSimulation) == 0)
	{
		LoadSimulation(childElem);
		childElem=childElem->getNextElementSibling();
	}

	return true;
}

void LoadNSProject::LoadNeuralNetwork(const XERCES_CPP_NAMESPACE::DOMElement* elem)
{
	const XERCES_CPP_NAMESPACE::DOMElement* deviceElem = elem->getFirstElementChild();
	if (!deviceElem)
		return;

	if (wcscmp(deviceElem->getLocalName(), elem::szFileDevice) == 0)
	{
		std::string nn_filename = util::StringUtil::WideToMultiByte(deviceElem->getAttribute(att::szFileName));
		if (!nn_filename.empty())
		{
			std::string nn_path = m_proj_dir;
			nn_path.append(nn_filename);
			m_project.OpenNetworkStructure(nn_path.c_str(), false);	// 나중에 LoadView를 할 것이기 때문에 여기에서 하진 않는다.
		}
	}
}

void LoadNSProject::LoadDataPreprocessor(const XERCES_CPP_NAMESPACE::DOMElement* elem)
{
	const XERCES_CPP_NAMESPACE::DOMElement* childElem = elem->getFirstElementChild();

	if (childElem && wcscmp(childElem->getLocalName(), elem::szIntegratedProvider) == 0)
	{
		m_provider.IntegratedProvider(true);
		LoadProvider(childElem, m_provider.GetPredictProvider());
	}
	else
	{
		if (childElem && wcscmp(childElem->getLocalName(), elem::szLearnProvider) == 0)
		{
			LoadProvider(childElem, *m_provider.GetLearnProvider());

			childElem = childElem->getNextElementSibling();
		}
		if (childElem && wcscmp(childElem->getLocalName(), elem::szPredictProvider) == 0)
		{
			LoadProvider(childElem, m_provider.GetPredictProvider());
		}
	}
}

void LoadNSProject::LoadProvider(const XERCES_CPP_NAMESPACE::DOMElement* elem, dp::model::DataProviderModel& provider)
{
	const XERCES_CPP_NAMESPACE::DOMElement* childElem = elem->getFirstElementChild();

	if (childElem && wcscmp(childElem->getLocalName(), elem::szDataReaderList) == 0)
	{
		const XERCES_CPP_NAMESPACE::DOMElement* readerElem = childElem->getFirstElementChild();
		for (; readerElem != NULL; readerElem = readerElem->getNextElementSibling())
			LoadDataReader(readerElem, provider);

		childElem = childElem->getNextElementSibling();
	}
	if (childElem && wcscmp(childElem->getLocalName(), elem::szProducerList) == 0)
	{
		const XERCES_CPP_NAMESPACE::DOMElement* producerElem = childElem->getFirstElementChild();
		for (; producerElem != NULL; producerElem = producerElem->getNextElementSibling())
			LoadProducer(producerElem, provider);

		childElem = childElem->getNextElementSibling();
	}
}

#include "NeuroData/model/BinaryReaderModel.h"
using namespace np::dp::model;

void LoadNSProject::LoadDataReader(const XERCES_CPP_NAMESPACE::DOMElement* elem, dp::model::DataProviderModel& provider)
{
	const wchar_t* uid_string = elem->getAttribute(att::szID);
	if (uid_string == NULL || wcslen(uid_string) == 0)
		return;

	neuro_u32 id=_wtoi(uid_string);
	std::string strName=util::StringUtil::WideToMultiByte(elem->getAttribute(att::szName));

	AbstractReaderModel* reader_model = NULL;

	std::wstring elemName = elem->getLocalName();
	if(elemName.compare(elem::szTextReader)==0)
	{
		TextReaderModel* text_reader = (TextReaderModel*)AbstractReaderModel::CreateInstance(provider, _reader_type::text, id);
		if (text_reader == NULL)
			return;
		reader_model = text_reader;

		text_reader->SetImportedSource(util::StringUtil::WideToMultiByte(elem->getAttribute(att::szImported)).c_str());

		text_reader->SetSkipFirstCount(_wtoi(elem->getAttribute(att::szStartSkip)));
		text_reader->SetReverse(wcscmp(elem->getAttribute(att::szReverse), att_value::boolean::szTrue) == 0);

		std::wstring strDelimiterType=elem->getAttribute(att::szDelimiterType);
		if (strDelimiterType.compare(att_value::delimiterType::szToken) == 0)
		{
			text_reader->ChangeDelimiterType(dp::_delimiter_type::token);

			dp::model::ExtTextColumnTokenDelimiter& token_delimiter = (dp::model::ExtTextColumnTokenDelimiter&)text_reader->GetDelimiter();
			token_delimiter.double_quote = wcscmp(elem->getAttribute(att::szDoubleQuotes), att_value::boolean::szFalse) != 0;
			token_delimiter.content_token = util::StringUtil::TransformTextToAscii(util::StringUtil::WideToMultiByte(elem->getAttribute(att::szContentToken)));

			StringToTokenVector(util::StringUtil::WideToMultiByte(elem->getAttribute(att::szColumnToken)), token_delimiter.column_token_vector);
		}
		else if (strDelimiterType.compare(att_value::delimiterType::szLength) == 0)
		{
			text_reader->ChangeDelimiterType(dp::_delimiter_type::length);

			dp::TextColumnLengthDelimiter& lengh_delimiter = (dp::TextColumnLengthDelimiter&)text_reader->GetDelimiter();

			const XERCES_CPP_NAMESPACE::DOMElement* srcItemElem = elem->getFirstElementChild();
			for (; srcItemElem != NULL; srcItemElem = srcItemElem->getNextElementSibling())
			{
				if (wcscmp(srcItemElem->getLocalName(), elem::szItem) != 0)
					continue;
				lengh_delimiter.lengh_vector.push_back(_wtoi(srcItemElem->getAttribute(att::szColumnLength)));
			}
		}
	}
	else if (elemName.compare(elem::szBinaryReader) == 0)
	{
		BinaryReaderModel* binary_reader = (BinaryReaderModel*)AbstractReaderModel::CreateInstance(provider, _reader_type::binary, id);
		if (binary_reader == NULL)
			return;
		reader_model = binary_reader;

		const XERCES_CPP_NAMESPACE::DOMElement* srcItemElem = elem->getFirstElementChild();
		for (; srcItemElem != NULL; srcItemElem = srcItemElem->getNextElementSibling())
		{
			if (wcscmp(srcItemElem->getLocalName(), elem::szItem) != 0)
				continue;

			_data_type_vector& type_vector = binary_reader->GetTypeVector();
			type_vector.push_back(ToDataType(srcItemElem->getAttribute(att::szColumnDataType)));
		}

		return;
	}

	if (reader_model)
	{
		const wchar_t* input_id_string = elem->getAttribute(att::szInputID);
		if (input_id_string && wcslen(input_id_string) != 0)
		{
			AbstractReaderModel* input = (AbstractReaderModel*) provider.GetDataModel(_wtoi(input_id_string));
			if (input && input->GetModelType() == _model_type::reader)
				reader_model->SetInput(input);

		}
		provider.AddReaderModel(reader_model);
	}
}

void LoadNSProject::LoadProducer(const XERCES_CPP_NAMESPACE::DOMElement* elem, dp::model::DataProviderModel& provider)
{
	const wchar_t* uid_string = elem->getAttribute(att::szID);
	if (uid_string == NULL || wcslen(uid_string) == 0)
		return;

	neuro_u32 uid = _wtoi(uid_string);
	neuro_u32 start_pos = _wtoi(elem->getAttribute(att::szStartPosition));

	AbstractProducerModel* producer_model = NULL;
	std::wstring elemName = elem->getLocalName();
	if (elemName.compare(elem::szNumericProducer) == 0)
	{
		NumericProducerModel* model = (NumericProducerModel*)AbstractProducerModel::CreateInstance(provider, dp::model::_producer_type::numeric, uid);
		if (model == NULL)
			return;

		const XERCES_CPP_NAMESPACE::DOMElement* itemListElem = elem->getFirstElementChild();
		if (wcscmp(itemListElem->getLocalName(), elem::szItemList) == 0)
		{
			const XERCES_CPP_NAMESPACE::DOMElement* indexElem = itemListElem->getFirstElementChild();
			for (; indexElem != NULL; indexElem = indexElem->getNextElementSibling())
			{
				if (wcscmp(indexElem->getLocalName(), elem::szItem) != 0)
					continue;

				neuro_u32 index = _wtoi(indexElem->getAttribute(att::szIndex));
				neuro_u32 ma = 1;
				const wchar_t* strMA = indexElem->getAttribute(att::szMA);
				if (strMA)
				{
					ma = _wtoi(strMA);
					if (ma == 0)
						ma = 1;
				}
				model->InsertSourceColumn(index, ma);
			}
		}
		model->SetLabelOutCount(_wtoi(elem->getAttribute(att::szOnehotSize)));

		producer_model = model;
	}
	else if (elemName.compare(elem::szImageProducer) == 0)
	{
		ImageFileProducerModel* model = (ImageFileProducerModel*)AbstractProducerModel::CreateInstance(provider, dp::model::_producer_type::image_file, uid);
		if (model == NULL)
			return;

		_NEURO_INPUT_IMAGEFILE_INFO info;
		info.sz.width = _wtoi(elem->getAttribute(att::szWidth));
		info.sz.height = _wtoi(elem->getAttribute(att::szHeight));

		const wchar_t* color_type = elem->getAttribute(att::szScaleType);
		if (wcscmp(color_type, att_value::imgScaleType::szMono) == 0)
			info.color_type = _color_type::mono;
		else
			info.color_type = _color_type::rgb;

		model->SetDefinition(info);
		producer_model = model;
	}
	else if (elemName.compare(elem::szWindllProducer) == 0)
	{
//		producer_model = new dp::WindllProducerModel;
	}
	else if (elemName.compare(elem::szMnistImgProducer) == 0)
	{
		producer_model = AbstractProducerModel::CreateInstance(provider, dp::model::_producer_type::mnist_img, uid);
	}
	else if (elemName.compare(elem::szMnistLabelProducer) == 0)
	{
		producer_model = AbstractProducerModel::CreateInstance(provider, dp::model::_producer_type::mnist_label, uid);
	}
	else if (elemName.compare(elem::szNLPProducer) == 0)
	{
		NlpProducerModel* model = (NlpProducerModel*)AbstractProducerModel::CreateInstance(provider, dp::model::_producer_type::nlp, uid);
		if (model == NULL)
			return;

		std::string mecap_rc_path;
		std::string w2v_path;

		const XERCES_CPP_NAMESPACE::DOMElement* childElem = elem->getFirstElementChild();
		if (wcscmp(childElem->getLocalName(), elem::szMecab) == 0)
		{
			mecap_rc_path = util::FileUtil::GetFilePath<char>("", util::StringUtil::WideToMultiByte(childElem->getAttribute(att::szFileName)).c_str());

			childElem = childElem->getNextElementSibling();
		}
		if (wcscmp(childElem->getLocalName(), elem::szW2V) == 0)
		{
			w2v_path = util::FileUtil::GetFilePath<char>("", util::StringUtil::WideToMultiByte(childElem->getAttribute(att::szFileName)).c_str());

			childElem = childElem->getNextElementSibling();
		}

		if (wcscmp(childElem->getLocalName(), elem::szItemList) == 0)
		{
			const XERCES_CPP_NAMESPACE::DOMElement* indexElem = childElem->getFirstElementChild();

			for (; indexElem != NULL; indexElem = indexElem->getNextElementSibling())
			{
				if (wcscmp(indexElem->getLocalName(), elem::szItem) != 0)
					continue;

				model->InsertSourceColumn(_wtoi(indexElem->getAttribute(att::szIndex)));
			}
		}

		bool use_morpheme_parser = wcscmp(elem->getAttribute(att::szUseMorphemeParser), att_value::boolean::szFalse) != 0;
		bool use_morpheme_vector = wcscmp(elem->getAttribute(att::szUseMorphemeTypeVector), att_value::boolean::szTrue) == 0;

		bool parse_sentence = wcscmp(elem->getAttribute(att::szParseSentence), att_value::boolean::szFalse) != 0;
		neuro_u32 max_word = _wtoi(elem->getAttribute(att::szMaxWord));
		neuro_u32 max_sentence = _wtoi(elem->getAttribute(att::szMaxSentence));
		neuro_u32 max_word_in_sentence = _wtoi(elem->getAttribute(att::szMaxWordPerSentence));

		bool is_wordvector_norm = wcscmp(elem->getAttribute(att::szWordVectorNom), att_value::boolean::szTrue) == 0;
		model->SetNLPInfo(mecap_rc_path.c_str(), w2v_path.c_str()
			, use_morpheme_parser, use_morpheme_vector, is_wordvector_norm, parse_sentence, max_word, max_sentence, max_word_in_sentence);

		producer_model = model;
	}
	else if (elemName.compare(elem::szIncreasePredictProducer) == 0)
	{
		IncreasePredictProducerModel* model = (IncreasePredictProducerModel*)AbstractProducerModel::CreateInstance(provider, dp::model::_producer_type::increase_predict, uid);
		if (model == NULL)
			return;

		model->SetPredictDistance(_wtoi(elem->getAttribute(att::szPredictDistance)));

		model->SetSourceColumn(_wtoi(elem->getAttribute(att::szIndex)));
		model->SetMovingAvarage(max(1, _wtoi(elem->getAttribute(att::szMA))));

		std::wstring preprocessor = elem->getAttribute(att::szPreprocessType);
		if (preprocessor.compare(att_value::classRangePreprocessType::szIncrease)==0)
			model->SetPredictType(_increase_predict_type::value);
		else if (preprocessor.compare(att_value::classRangePreprocessType::szIncreaseRate) == 0)
			model->SetPredictType(_increase_predict_type::rate);

		_predict_range_vector ranges;

		const XERCES_CPP_NAMESPACE::DOMElement* rangeElem = elem->getFirstElementChild();
		for (; rangeElem != NULL; rangeElem = rangeElem->getNextElementSibling())
		{
			if (wcscmp(rangeElem->getLocalName(), elem::szRange) != 0)
				continue;

			_PREDICT_RANGE_INFO model;
			model.value = neuron_value(_wtof(rangeElem->getAttribute(att::szValue)));

			std::wstring ineuality=rangeElem->getAttribute(att::szInequality);
			if (ineuality.compare(L"<")==0)
				model.ineuality = _inequality_type::less;
			else if (ineuality.compare(L"<=") == 0)
				model.ineuality = _inequality_type::equal_less;
			else if (ineuality.compare(L">=") == 0)
				model.ineuality = _inequality_type::equal_greater;
			else if (ineuality.compare(L">") == 0)
				model.ineuality = _inequality_type::greater;

			ranges.push_back(model);
		}
		model->SetRanges(ranges);

		producer_model = model;
	}
	else if (elemName.compare(elem::szAddinProducer) == 0)
	{
	}

	if (!producer_model)
		return;

	if (!provider.AddProducerModel(producer_model))
	{
		delete producer_model;
		return;
	}

	const wchar_t* input_id_string = elem->getAttribute(att::szInputID);
	if (input_id_string && wcslen(input_id_string) != 0)
	{
		AbstractReaderModel* input = (AbstractReaderModel*)provider.GetDataModel(_wtoi(input_id_string));
		if (input && input->GetModelType() == _model_type::reader)
			producer_model->SetInput(input);
	}

	producer_model->SetStartPosition(start_pos);

	const XERCES_CPP_NAMESPACE::DOMElement* bindingListElem = elem->getFirstElementChild();
	if (wcscmp(bindingListElem->getLocalName(), elem::szBindingList) == 0)
	{
		network::NeuralNetwork* network = m_project.GetNSManager().GetNetwork();

		const XERCES_CPP_NAMESPACE::DOMElement* binding_elem = bindingListElem->getFirstElementChild();
		for (; binding_elem != NULL; binding_elem = binding_elem->getNextElementSibling())
		{
			if (wcscmp(binding_elem->getLocalName(), elem::szBinding) == 0)
			{
				neuro_u32 binding_id = _wtoi(binding_elem->getAttribute(att::szID));

				network::AbstractLayer* layer = network->FindLayer(binding_id);
				if (layer && (layer->GetLayerType()==network::_layer_type::input || layer->GetLayerType()==network::_layer_type::output))
					layer->AddBinding(producer_model);
			}
		}
	}
}

void LoadNSProject::LoadSimulation(const XERCES_CPP_NAMESPACE::DOMElement* elem)
{
	const XERCES_CPP_NAMESPACE::DOMElement* childElem=elem->getFirstElementChild();
	if(childElem && wcscmp(childElem->getLocalName(), elem::szResultViewList)==0)
	{
		LoadSimResultViewList(childElem);
		childElem = childElem->getNextElementSibling();
	}

	if (childElem && wcscmp(childElem->getLocalName(), elem::szEnv) == 0)
	{
		LoadSimEnvElem(childElem);
		childElem = childElem->getNextElementSibling();
	}

	if (childElem && wcscmp(childElem->getLocalName(), elem::szTrainData) == 0)
	{
		LoadSimTrainData(childElem);
		childElem = childElem->getNextElementSibling();
	}

	if (childElem && wcscmp(childElem->getLocalName(), elem::szPredictData) == 0)
	{
		LoadSimPredictData(childElem);
		childElem = childElem->getNextElementSibling();
	}
}

void LoadNSProject::LoadSimResultViewList(const XERCES_CPP_NAMESPACE::DOMElement* elem)
{
	project::SimDefinition& sim_def=m_project.GetSimManager();

	const XERCES_CPP_NAMESPACE::DOMElement* viewElem = elem->getFirstElementChild();
	for (; viewElem != NULL; viewElem = viewElem->getNextElementSibling())
	{
		if (wcscmp(viewElem->getLocalName(), elem::szResultView) != 0)
			continue;

		_LAYER_DISPLAY_INFO info;
		info.type = project::_layer_display_type::none;
		std::wstring strType = viewElem->getAttribute(att::szType);
		for (neuro_u32 i = 0; i < _countof(project::layer_display_type_string); i++)
		{
			if (strType == project::layer_display_type_string[i])
			{
				info.type = (_layer_display_type)i;
				break;
			}
		}
		info.is_argmax_output = wcscmp(viewElem->getAttribute(att::szArgmaxOutput), att_value::boolean::szTrue) == 0;
		info.is_onehot_analysis_result = wcscmp(viewElem->getAttribute(att::szOnehotAnalysisResult), att_value::boolean::szTrue)==0;

		sim_def.SetLayerDisplayInfo(_wtoll(viewElem->getAttribute(att::szLayerUID)), info);
	}
}

void LoadNSProject::LoadSimEnvElem(const XERCES_CPP_NAMESPACE::DOMElement* elem)
{
	const XERCES_CPP_NAMESPACE::DOMElement* childElem = elem->getFirstElementChild();
	if (childElem && wcscmp(childElem->getLocalName(), elem::szTrainEnv) == 0)
	{
		SIM_TRAIN_ENV env;
		env.minibatch_size = _wtol(childElem->getAttribute(att::szMinibatchSize));
		env.useNdf = wcscmp(childElem->getAttribute(att::szUseNdf), att_value::boolean::szFalse) != 0;
		env.data_noising = wcscmp(childElem->getAttribute(att::szDataNoising), att_value::boolean::szTrue) == 0;

		const XERCES_CPP_NAMESPACE::DOMElement* envElem = childElem->getFirstElementChild();
		for (; envElem != NULL; envElem = envElem->getNextElementSibling())
		{
			std::wstring name = envElem->getLocalName();
			if (name.compare(elem::szEndCondition) == 0)
			{
				env.max_epoch = _wtoll(envElem->getAttribute(att::szMaxEpoch));

				const XERCES_CPP_NAMESPACE::DOMElement* condElem = envElem->getFirstElementChild();
				if (condElem && wcscmp(condElem->getLocalName(), elem::szBelowError) == 0)
				{
					env.is_end_below_error = true;
					env.close_error = static_cast<neuron_error>(_wtof(envElem->getAttribute(att::szError)));

					condElem = envElem->getNextElementSibling();
				}
			}
			else if (name.compare(elem::szAnalyze) == 0)
			{
				env.bTestAfterLearn = wcscmp(envElem->getAttribute(att::szTestAfterLearn), att_value::boolean::szFalse) != 0;
				env.bAnalyzeLossHistory = wcscmp(envElem->getAttribute(att::szLossHistory), att_value::boolean::szTrue) == 0;
				env.bAnalyzeArgmaxAccuracy = wcscmp(envElem->getAttribute(att::szArgmaxAccuracy), att_value::boolean::szTrue) == 0;
			}
			else if (name.compare(elem::szDisplay) == 0)
			{
				env.display_period_sample = _wtol(envElem->getAttribute(att::szPeriodSample));
			}
		}
		m_project.GetSimManager().SetTrainEnv(env);

		childElem = childElem->getNextElementSibling();
	}
	if (childElem && wcscmp(childElem->getLocalName(), elem::szRunEnv) == 0)
	{
		SIM_RUN_ENV env;
		env.minibatch_size = _wtol(childElem->getAttribute(att::szMinibatchSize));

		const XERCES_CPP_NAMESPACE::DOMElement* runElem = childElem->getFirstElementChild();
		if (runElem && wcscmp(runElem->getLocalName(), elem::szDisplay) == 0)
		{
			env.display_period_sample = _wtol(runElem->getAttribute(att::szPeriodSample));

			runElem = runElem->getNextElementSibling();
		}

		m_project.GetSimManager().SetRunEnv(env);

		childElem = childElem->getNextElementSibling();
	}
}

void LoadNSProject::LoadSimTrainData(const XERCES_CPP_NAMESPACE::DOMElement* elem)
{
	const XERCES_CPP_NAMESPACE::DOMElement* trainElem = elem->getFirstElementChild();

	project::SimDefinition& sim_def = m_project.GetSimManager();

	for (; trainElem != NULL; trainElem = trainElem->getNextElementSibling())
	{
		if (wcscmp(trainElem->getLocalName(), elem::szTrainLearn) == 0)
			LoadSimData(trainElem, sim_def.GetLastLearnData());
		else if (wcscmp(trainElem->getLocalName(), elem::szTrainTest) == 0)
			LoadSimData(trainElem, sim_def.GetLastTestData());
	}
}

void LoadNSProject::LoadSimPredictData(const XERCES_CPP_NAMESPACE::DOMElement* elem)
{
	const XERCES_CPP_NAMESPACE::DOMElement* childElem = elem->getFirstElementChild();

	if (childElem && wcscmp(childElem->getLocalName(), elem::szPredictInput) == 0)
	{
		LoadSimData(childElem, m_project.GetSimManager().GetLastPredictData());
		childElem = childElem->getNextElementSibling();
	}

	if (childElem && wcscmp(childElem->getLocalName(), elem::szPredictOutput) == 0)
	{
		dp::_STREAM_WRITE_INFO& write_info = m_project.GetSimManager().GetLastPredictOutputWriteInfo();
		write_info.no_type_prefix = util::StringUtil::WideToMultiByte(childElem->getAttribute(att::szOutputNoPrefix));

		const XERCES_CPP_NAMESPACE::DOMElement* fileElem = childElem->getFirstElementChild();
		if (fileElem && wcscmp(fileElem->getLocalName(), elem::szFile) == 0)
			m_project.GetSimManager().SetLastPredictOutputPath(fileElem->getAttribute(att::szFilePath));

		childElem = childElem->getNextElementSibling();
	}
}

void LoadNSProject::LoadSimData(const XERCES_CPP_NAMESPACE::DOMElement* elem, dp::preprocessor::_uid_datanames_map& source_map)
{
	source_map.clear();

	const XERCES_CPP_NAMESPACE::DOMElement* dataElem = elem->getFirstElementChild();
	for (; dataElem != NULL; dataElem = dataElem->getNextElementSibling())
	{
		if (wcscmp(dataElem->getLocalName(), elem::szItem) != 0)
			continue;

		const wchar_t* id_string = dataElem->getAttribute(att::szID);
		if (id_string == NULL)
			continue;

		neuro_u32 uid = _wtoi(id_string);
		DataSourceNameVector<char>& source_name_vector = source_map[uid];

		const XERCES_CPP_NAMESPACE::DOMElement* fileElem = dataElem->getFirstElementChild();
		for (; fileElem != NULL; fileElem = fileElem->getNextElementSibling())
		{
			if (wcscmp(fileElem->getLocalName(), elem::szFile) != 0)
				continue;

			std::string path = util::FileUtil::GetFilePath<char>(m_proj_dir, util::StringUtil::WideToMultiByte(fileElem->getAttribute(att::szFilePath)));
			if (!path.empty())
				source_name_vector.AddPath(path.c_str());
		}
		if (source_name_vector.GetCount()==0)
			source_map.erase(uid);
	}
}
