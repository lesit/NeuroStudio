#pragma once

#include "AbstractPreprocessorModel.h"

namespace np
{
	namespace dp
	{
		namespace model
		{
			static const wchar_t* _reader_type_string[] = { L"Text", L"Binary"/*, L"Database"*/};
			static _reader_type ToReaderType(const wchar_t* name)
			{
				for (neuro_u8 type = 0; type < _countof(_reader_type_string); type++)
				if (wcscmp(name, _reader_type_string[type]) == 0)
					return (_reader_type)type;

				return _reader_type::unknown;
			}
			static const wchar_t* ToReaderString(_reader_type type)
			{
				if ((neuro_u32)type >= _countof(_reader_type_string))
					return L"";

				return _reader_type_string[(neuro_u32)type];
			}

			enum class _move_direct{ first, last, prev, next };
			class AbstractReaderModel : public AbstractPreprocessorModel
			{
			public:
				static AbstractReaderModel* CreateInstance(DataProviderModel& provider, _reader_type type, neuro_u32 uid);

				virtual ~AbstractReaderModel() {}

				_model_type GetModelType() const { return _model_type::reader; }

				// has_input_reader_status HasInputReaderStatus() const override { return _has_input_reader_status::yes; }
				// 나중에 database reader를 만들면 yes로 하자
				_has_input_reader_status HasInputReaderStatus() const override { return _has_input_reader_status::no; }

				static std::unordered_set<_reader_type> GetAvailableInputReaderTypeSet(_reader_type owner_type)
				{
					switch (owner_type)
					{
					case _reader_type::binary:
					case _reader_type::text:
						return{/*_reader_type::database*/ };
/*					case _reader_type::database:
						return{};*/
					}
					return{};
				}

				std::unordered_set<_reader_type> GetAvailableInputReaderTypeSet() const override
				{
					return GetAvailableInputReaderTypeSet(GetReaderType());
				}

				inline bool AvailableChangeType(_reader_type type) const
				{
					if (m_input)	// 입력이 있을 경우 새로운 type이 가능한 입력 type인지 검사
					{
						std::unordered_set<_reader_type> available_set = GetAvailableInputReaderTypeSet(type);
						if (available_set.find(m_input->GetReaderType()) == available_set.end())
							return false;
					}

					// 출력에 대해 새로운 type을 입력으로 가질수 있는지 검사
					for (_preprocessor_model_set::const_iterator it = m_output_set.begin(); it != m_output_set.end(); it++)
					{
						if (!(*it)->AvailableInput(type))
							return false;
					}
					return true;
				}
				void GetAvailableChangeTypes(std::vector<_reader_type>& type_vector) const
				{
					for (neuro_u32 i = 0; i < 2; i++)
					{
						if (AvailableChangeType((_reader_type)i))
							type_vector.push_back((_reader_type)i);
					}
				}

				void ChangedInputProperty() override {}	// 나중에 이중 reader를 사용하는 db 같은거 생길때 구현하자
				void ChangedProperty() override
				{
					_preprocessor_model_set::iterator it = m_output_set.begin();
					for (; it != m_output_set.end(); it++)
						(*it)->ChangedInputProperty();
				}

				virtual _reader_type GetReaderType() const = 0;

				virtual bool CopyFrom(const AbstractReaderModel& src)
				{
					return src.GetReaderType() != GetReaderType();
				}

				virtual void SetColumnCount(neuro_u32 count) = 0;
				virtual neuro_u32 GetColumnCount() const = 0;

				void GetUsingSourceIndexSet(_u32_set& index_set) const override;

			protected:
				AbstractReaderModel(DataProviderModel& provider, neuro_u32 uid)
					: AbstractPreprocessorModel(provider, uid)
				{}
			};

			typedef std::vector<AbstractReaderModel*> _reader_model_vector;
		}
	}
}
