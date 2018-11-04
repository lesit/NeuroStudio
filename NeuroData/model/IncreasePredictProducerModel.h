#pragma once

#include "NumericProducerModel.h"

namespace np
{
	namespace dp
	{
		namespace model
		{
			enum class _increase_predict_type { value, rate };
			static const wchar_t* _increase_predict_type_string[] = { L"value", L"rate" };
			static const wchar_t* ToString(_increase_predict_type type)
			{
				if ((neuro_u32)type >= _countof(_increase_predict_type_string))
					return L"";
				return _increase_predict_type_string[(neuro_u32)type];
			}
			static _increase_predict_type ToIncreasePredictType(const wchar_t* name)
			{
				for (neuro_u32 i = 0; i < _countof(_increase_predict_type_string); i++)
				{
					if (wcscmp(_increase_predict_type_string[i], name) == 0)
						return (_increase_predict_type)i;
				}
				return _increase_predict_type::rate;
			}

			enum class _inequality_type { less, equal_less, greater, equal_greater };
			static const wchar_t* _inequality_type_string[] = { L"<", L"<=", L">", L">=" };
			static const wchar_t* ToString(_inequality_type type)
			{
				if ((neuro_u32)type >= _countof(_inequality_type_string))
					return L"";
				return _inequality_type_string[(neuro_u32)type];

			}
			static _inequality_type ToInequalityType(const wchar_t* name)
			{
				for (neuro_u32 i = 0; i < _countof(_inequality_type_string); i++)
				{
					if (wcscmp(_inequality_type_string[i], name) == 0)
						return (_inequality_type)i;
				}
				return _inequality_type::greater;
			}
			
			struct _PREDICT_RANGE_INFO
			{
				neuron_value value;
				_inequality_type ineuality;
			};
			typedef std::vector<_PREDICT_RANGE_INFO> _predict_range_vector;

			class IncreasePredictProducerModel : public AbstractProducerModel
			{
			public:
				IncreasePredictProducerModel(DataProviderModel& provider, neuro_u32 uid)
					: AbstractProducerModel(provider, uid)
				{
					m_src_column = 0;
					m_moving_average = 1;

					m_predict_distance = 0;

					m_increase_predict_type = _increase_predict_type::rate;
				}
				virtual ~IncreasePredictProducerModel() {}

				virtual _producer_type GetProducerType() const override { return _producer_type::increase_predict; }

				_has_input_reader_status HasInputReaderStatus() const override { return _has_input_reader_status::must; }
				virtual bool AvailableToInputLayer() const { return false; }

				virtual tensor::DataShape GetDataShape() const override
				{
					return tensor::DataShape({ (neuro_u32)m_range_vector.size() });;
				}

				void SetPredictDistance(neuro_u32 distance)
				{
					m_predict_distance = distance;
				}

				neuro_u32 GetPredictDistance() const { return m_predict_distance; }

				void SetSourceColumn(neuro_u32 column) { m_src_column = column; }
				neuro_u32 GetSourceColumn() const { return m_src_column; }

				void SetMovingAvarage(neuro_u32 avr) { m_moving_average = avr; }
				neuro_u32 GetMovingAvarage() const { return m_moving_average; }

				void SetPredictType(_increase_predict_type type) { m_increase_predict_type = type;}
				_increase_predict_type GetPredictType() const { return m_increase_predict_type; }

				void SetRanges(const _predict_range_vector& ranges)	
				{
					m_range_vector = ranges;
					ChangedProperty();
				}
				const _predict_range_vector& GetRanges() const { return m_range_vector; }

				NumericProducerModel* CreateSourceModel() const
				{
					NumericProducerModel* ret = new NumericProducerModel(m_provider, 0);
					ret->InsertSourceColumn(m_src_column, m_moving_average);
					ret->SetStartPosition(GetStartPosition());
					return ret;
				}

				NumericProducerModel* CreatePredictModel() const
				{
					NumericProducerModel* ret = new NumericProducerModel(m_provider, 1);
					ret->InsertSourceColumn(m_src_column, m_moving_average);
					ret->SetStartPosition(ret->GetStartPosition() + m_predict_distance);
					return ret;
				}
			private:
				neuro_u32 m_src_column;
				neuro_u32 m_moving_average;

				neuro_u32 m_predict_distance;

				_increase_predict_type m_increase_predict_type;

				_predict_range_vector m_range_vector;
			};
		}
	}
}
