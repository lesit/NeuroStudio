#pragma once

#include <xercesc/util/XercesDefs.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/dom/DOMElement.hpp>

#include "NeuroData/model/PreprocessorModelIncludeHeader.h"

namespace np
{
	namespace project
	{
		namespace xml
		{
			namespace elem
			{
				static const wchar_t* szRoot			= L"np";
				
				static const wchar_t* szNeuralNetwork = L"NeuralNetwork";
				static const wchar_t* szFileDevice = L"device.file";

				static const wchar_t* szNLPInfo = L"nlp_info";
				static const wchar_t* szMecab = L"mecab";
				static const wchar_t* szW2V = L"w2v";

				static const wchar_t* szDataPreprocessor = L"data.preprocessor";
				static const wchar_t* szIntegratedProvider = L"provider.integrated";
				static const wchar_t* szLearnProvider = L"provider.learn";
				static const wchar_t* szPredictProvider = L"provider.predict";
				// in provider.train, provider.predict
				static const wchar_t* szDataReaderList = L"reader.list";
				static const wchar_t* szProducerList = L"producer.list";

				// in szDataReaderList
				static const wchar_t* szBinaryReader = L"reader.byte";
				static const wchar_t* szTextReader = L"reader.text";
				static const wchar_t* szDBReader = L"reader.db";

				// in szProducerList
				static const wchar_t* szNumericProducer	= L"producer.numeric";

				static const wchar_t* szImageProducer = L"producer.image";

				static const wchar_t* szJavascriptProducer		= L"producer.javascript";
				static const wchar_t* szParam = L"param";
				static const wchar_t* szJSBody = L"jsbody";

				static const wchar_t* szWindllProducer = L"producer.windll";
				static const wchar_t* szAddinProducer = L"producer.addin";

				static const wchar_t* szMnistImgProducer = L"producer.mnist.image";
				static const wchar_t* szMnistLabelProducer = L"producer.mnist.label";

				static const wchar_t* szNLPProducer = L"producer.nlp";

				static const wchar_t* szIncreasePredictProducer = L"producer.increase_predict";
				static const wchar_t* szRange = L"range";

				// in producer
				static const wchar_t* szBindingList = L"binding.list";

				// in producer/display
				static const wchar_t* szBinding = L"binding";

				// in numeric/nlp
				static const wchar_t* szItemList = L"item.list";
				static const wchar_t* szItem = L"item";

				static const wchar_t* szSimulation		= L"simulation";
				static const wchar_t* szResultViewList = L"resultView.list";
				static const wchar_t* szResultView = L"resultView";

				static const wchar_t* szGraph			= L"graph";
				static const wchar_t* szGraphView		= L"view";
				static const wchar_t* szGraphLine		= L"line";
				static const wchar_t* szHorzLabel		= L"horzLabel";

				static const wchar_t* szListItem		= L"item";

				static const wchar_t* szEnv = L"env";
				static const wchar_t* szTrainEnv = L"env.train";
				static const wchar_t* szRunEnv = L"env.run";

				static const wchar_t* szEndCondition = L"end.condition";
				static const wchar_t* szBelowError = L"below.error";

				static const wchar_t* szDisplay = L"display";

				static const wchar_t* szAnalyze = L"analyze";

				static const wchar_t* szTrainData = L"train.data";
				static const wchar_t* szTrainLearn = L"train.learn";
				static const wchar_t* szTrainTest = L"train.test";
				static const wchar_t* szPredictData = L"predict.data";
				static const wchar_t* szPredictInput = L"predict.input";
				static const wchar_t* szPredictOutput = L"predict.output";
				static const wchar_t* szFile = L"file";
			}

			namespace att
			{
				static const wchar_t* szID				= L"id";
				static const wchar_t* szName			= L"name";

				static const wchar_t* szInputID = L"input.id";

				static const wchar_t* szValue = L"value";
				static const wchar_t* szInequality = L"inequality";

				static const wchar_t* szType			= L"type";

				static const wchar_t* szOnehotSize = L"onehot.size";

				static const wchar_t* szStartPosition = L"start.pos";
				static const wchar_t* szStartSkip = L"start.skip";
				static const wchar_t* szReverse = L"reverse";

				static const wchar_t* szDelimiterType	= L"delimiter.type";
				static const wchar_t* szColumnToken = L"column.token";

				static const wchar_t* szColumnCount = L"column.count";
				static const wchar_t* szColumnLength = L"column.length";
				static const wchar_t* szColumnDataType = L"column.datatype";

				static const wchar_t* szImported = L"imported";
				static const wchar_t* szContentToken = L"content.token";
				static const wchar_t* szDoubleQuotes = L"doubleQuotes";

				static const wchar_t* szFilePath = L"filepath";
				static const wchar_t* szFileName = L"filename";

				static const wchar_t* szOutputNoPrefix = L"output.no.prefix";

				static const wchar_t* szIndex			= L"index";

				static const wchar_t* szMA				= L"ma";

				// Class Range Producer
				static const wchar_t* szPredictDistance = L"predict.distance";
				static const wchar_t* szPreprocessType = L"preprocess.type";

				static const wchar_t* szUseMorphemeParser	= L"use_morpheme_parser";
				static const wchar_t* szUseMorphemeTypeVector	= L"use_morpheme_vector";
				static const wchar_t* szWordVectorNom		= L"wordvector.norm";
				static const wchar_t* szParseSentence		= L"sentence.parse";
				static const wchar_t* szMaxWord				= L"word.max";
				static const wchar_t* szMaxSentence			= L"sentence.max";
				static const wchar_t* szMaxWordPerSentence	= L"sentence.word.max";

				static const wchar_t* szScaleType		= L"scale.type";
				static const wchar_t* szScaleFirst		= L"scale.first";
				static const wchar_t* szScaleSecond		= L"scale.second";

				static const wchar_t* szStreamID		= L"stream.id";
				static const wchar_t* szLayerUID = L"layer.uid";

				static const wchar_t* szWidth			= L"width";
				static const wchar_t* szHeight			= L"height";

				static const wchar_t* szColor			= L"color";

				static const wchar_t* szHasArgmaxOutput = L"argmax";

				static const wchar_t* szUseNdf = L"ndf.use";
				static const wchar_t* szDataNoising = L"data.noising";

				static const wchar_t* szMinibatchSize = L"minibatch.size";

				// in szEndCondition
				static const wchar_t* szMaxEpoch = L"epoch.max";
				// in szBelowError
				static const wchar_t* szError = L"error";

				// in szAnalyze
				static const wchar_t* szLossHistory = L"loss_history";
				static const wchar_t* szTestAfterLearn = L"test_after_learn";
				static const wchar_t* szArgmaxAccuracy = L"argmax_accuracy";

				// in szDisplay
				static const wchar_t* szPeriodSample = L"period.sample";

				static const wchar_t* szArgmaxOutput = L"argmax.output";
				static const wchar_t* szOnehotAnalysisResult = L"onehot.analysis.result";
			}

			namespace att_value
			{
				namespace boolean
				{
					static const wchar_t* szTrue		= L"true";
					static const wchar_t* szFalse		= L"false";
				}

				namespace dataType
				{
					static const wchar_t* szDataTypeName[]	= {L"integer", L"int64",  L"float", L"percent", L"time", L"string", NULL};
				}

				namespace delimiterType
				{
					static const wchar_t* szToken	= L"token";
					static const wchar_t* szLength		= L"length";
				}

				namespace scaleType
				{
					static const wchar_t* szNone		= L"none";
					static const wchar_t* szMinMax		= L"minmax";
					static const wchar_t* szBoolean		= L"boolean";
					static const wchar_t* szInt32		= L"int32";
				}

				namespace imgScaleType
				{
					static const wchar_t* szMono = L"mono";
					static const wchar_t* szRGB = L"rgb";
				}

				namespace targetType
				{
					static const wchar_t* szTargetStream	= L"target";
					static const wchar_t* szJavaScript		= L"js";
					static const wchar_t* szJava			= L"java";
					static const wchar_t* szDll				= L"dll";
				}

				namespace targetRetvalue
				{
					static const wchar_t* szError = L"error";
					static const wchar_t* szTarget = L"target";
				}

				namespace classRangePreprocessType
				{
					static const wchar_t* szIncrease = L"increase";
					static const wchar_t* szIncreaseRate = L"increase.rate";
				}

				namespace simDisplayDelegateType
				{
					static const wchar_t* szSelect = L"select";
					static const wchar_t* szWorst = L"worst";
					static const wchar_t* szBest = L"best";
				}

				namespace simViewType
				{
					static const wchar_t* szImage		= L"image";
					static const wchar_t* szGraph		= L"graph";
					static const wchar_t* szList		= L"list";
				}

				namespace shapeType
				{
					static const wchar_t* szDot			= L"dot";
					static const wchar_t* szBar = L"bar";
					static const wchar_t* szLine = L"line";
				}

				namespace streamType
				{
					static const wchar_t* szFile		= L"file";
				}
			}

			namespace namespace_uri
			{
				static const wchar_t* szNamespaceUri	= L"np:system";
			}

			class NSProjectXML
			{
			public:
				NSProjectXML();
				virtual ~NSProjectXML();

				const XERCES_CPP_NAMESPACE::DOMDocument* GetDOMDocument() const {return m_document;}

				virtual XERCES_CPP_NAMESPACE::DOMDocument* LoadXML(const wchar_t* strXMLFilePath);

				XERCES_CPP_NAMESPACE::DOMDocument* CreateDocument();

				bool SaveXML(const wchar_t* strXMLFilePath);

				static unsigned int GetAttributeType(LPCTSTR strType, const wchar_t*const* szTypeArray, unsigned int nUnknownType, unsigned int nDefaultType);
				static unsigned int GetAttributeType(const XERCES_CPP_NAMESPACE::DOMElement& elem, LPCTSTR strAttName, const wchar_t*const* szTypeArray, unsigned int nUnknownType, unsigned int nDefaultType=-1);

				XERCES_CPP_NAMESPACE::DOMDocument* m_document;
			};
		}
	}
}
