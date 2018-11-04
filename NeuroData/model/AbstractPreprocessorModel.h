#pragma once

#include "common.h"

namespace np
{
	namespace dp	// Neuro Data Preprocessing
	{
		namespace model
		{
			class DataProviderModel;

			enum class _input_source_type { none, db, mnist_img_file, mnist_label_file, imagefile, binaryfile, textfile, dll };	// simulation에서 사용하기 위해서

			static const char* _source_type_string[] = { "", "Database", "Mnist file", "Image file", "Binary file", "Text file", "dynamic liabrary"};
			static _input_source_type ToSourceType(const char* name)
			{
				for (neuro_u8 type = 0; type < _countof(_source_type_string); type++)
					if (strcmp(name, _source_type_string[type]) == 0)
						return (_input_source_type)type;

				return _input_source_type::binaryfile;
			}
			static const char* ToSourceString(_input_source_type type)
			{
				if ((neuro_u32)type >= _countof(_source_type_string))
					return "";
				return _source_type_string[(neuro_u32)type];
			}

			enum class _model_type { reader, producer };
			enum class _reader_type { text, binary, /*database,*/ unknown };

			class AbstractReaderModel;

			class AbstractPreprocessorModel;
			typedef std::unordered_set<AbstractPreprocessorModel*> _preprocessor_model_set;
			class AbstractPreprocessorModel
			{
			public:
				AbstractPreprocessorModel(DataProviderModel& provider, neuro_u32 _uid);
				virtual ~AbstractPreprocessorModel();

				DataProviderModel& GetProvider() { return m_provider; }
				const DataProviderModel& GetProvider() const { return m_provider; }

				bool IsInPredictProvider() const;

				virtual _model_type GetModelType() const = 0;

				const neuro_u32 uid;

				virtual _input_source_type GetInputSourceType() const;

				bool AvailableAttachDeviceInput() const;

				enum class _has_input_reader_status{no, yes, must};
				virtual _has_input_reader_status HasInputReaderStatus() const { return _has_input_reader_status::yes; }

				std::vector<_reader_type> GetAvailableInputReaderTypeVector() const
				{
					std::unordered_set<_reader_type> available_set = GetAvailableInputReaderTypeSet();

					std::vector<_reader_type> ret;
					for (neuro_u32 i = 0; i < (neuro_u32)_reader_type::unknown; i++)
					{
						if (available_set.find((_reader_type)i) != available_set.end())
							ret.push_back((_reader_type)i);
					}
					return ret;
				}
				virtual std::unordered_set<_reader_type> GetAvailableInputReaderTypeSet() const
				{
					return{};
				}

				bool AvailableInput(_reader_type reader_type) const;
				void SetInput(AbstractReaderModel* input);

				const AbstractReaderModel* GetInput() const { return m_input; }
				AbstractReaderModel* GetInput() { return m_input; }

				const _preprocessor_model_set& GetOutputSet() const { return m_output_set; }
				_preprocessor_model_set& GetOutputSet() { return m_output_set; }

				// 현재는 Numeric, NLP, IncreasePredict producer 에서 사용
				virtual void ChangedInputProperty()
				{
					ChangedProperty();
				}
				virtual void ChangedProperty() = 0;

				virtual void GetUsingSourceIndexSet(_u32_set& index_set) const {}
			protected:
				DataProviderModel& m_provider;

				AbstractReaderModel* m_input;

				_preprocessor_model_set m_output_set;
			};
		}
	}
}
