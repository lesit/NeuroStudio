#pragma once

#include "common.h"

#include "NeuroData/StreamWriter.h"

namespace np
{
	using namespace dp::preprocessor;

	namespace project
	{
		struct SIM_TRAIN_ENV
		{
			SIM_TRAIN_ENV()
			{
				useNdf = true;
				data_noising = false;

				minibatch_size = 32;
				max_epoch = 1000;

				is_end_below_error = false;
				close_error = 0.0f;

				bTestAfterLearn = true;
				bAnalyzeArgmaxAccuracy = true;
				bAnalyzeLossHistory = true;

				display_period_sample = 1;
			}
			bool useNdf;
			bool data_noising;

			neuro_u32 minibatch_size;
			neuro_u64 max_epoch;

			bool is_end_below_error;
			neuron_error close_error;

			bool bTestAfterLearn;
			bool bAnalyzeArgmaxAccuracy;
			bool bAnalyzeLossHistory;

			neuro_u32 display_period_sample;
		};

		struct SIM_RUN_ENV
		{
			SIM_RUN_ENV()
			{
				minibatch_size = 32;

				display_period_sample = 1;
			}
			neuro_u32 minibatch_size;

			neuro_u32 display_period_sample;
		};
		class NeuroSystemManager;

		enum class _layer_display_type { image, list, none };
		static const wchar_t* layer_display_type_string[] = { L"image", L"list", L"none"};
		static const wchar_t* ToString(_layer_display_type type)
		{
			if ((int)type >= _countof(layer_display_type_string))
				return L"";

			return layer_display_type_string[(int)type];
		}

		struct _LAYER_DISPLAY_INFO
		{
			_LAYER_DISPLAY_INFO()
			{
				type = _layer_display_type::image;
				is_argmax_output = false;
				is_onehot_analysis_result = false;
			}
			_layer_display_type type;
			bool is_argmax_output;
			bool is_onehot_analysis_result;
		};
		typedef std::unordered_map<neuro_u32, _LAYER_DISPLAY_INFO> _layer_display_info_map;

		class SimDefinition
		{
		public:
			SimDefinition(NeuroSystemManager& stManager);
			virtual ~SimDefinition();

			bool IsEmpty() const;

			void SetLayerDisplayInfo(neuro_u32 layer_uid, const _LAYER_DISPLAY_INFO& info);
			_layer_display_info_map& GetLayerDisplayInfoMap() { return m_layer_display_info_map; }
			const _layer_display_info_map& GetLayerDisplayInfoMap() const { return m_layer_display_info_map; }

			NeuroSystemManager& GetNSManager(){return m_nsManager;}

			void SetTrainEnv(const SIM_TRAIN_ENV& env)
			{
				memcpy(&m_train_env, &env, sizeof(SIM_TRAIN_ENV));
			}

			const SIM_TRAIN_ENV& GetTrainEnv() const{
				return m_train_env;
			}

			void SetRunEnv(const SIM_RUN_ENV& env)
			{
				memcpy(&m_run_env, &env, sizeof(SIM_RUN_ENV));
			}

			const SIM_RUN_ENV& GetRunEnv() const{
				return m_run_env;
			}

			const _uid_datanames_map& GetLastLearnData() const { return m_last_learn_data_map; }
			_uid_datanames_map& GetLastLearnData() { return m_last_learn_data_map; }

			const _uid_datanames_map& GetLastTestData() const { return m_last_test_data_map; }
			_uid_datanames_map& GetLastTestData() { return m_last_test_data_map; }

			const _uid_datanames_map& GetLastPredictData() const { return m_last_predict_data_map;}
			_uid_datanames_map& GetLastPredictData() {	return m_last_predict_data_map;	}

			const std::wstring& GetLastPredictOutputPath() const
			{
				return m_last_predict_output_path;
			}

			void SetLastPredictOutputPath(const wchar_t* path)
			{
				m_last_predict_output_path = path;
			}

			dp::_STREAM_WRITE_INFO& GetLastPredictOutputWriteInfo()
			{
				return m_last_run_output_write_info;
			}

			const dp::_STREAM_WRITE_INFO& GetLastPredictOutputWriteInfo() const
			{
				return m_last_run_output_write_info;
			}
		private:
			NeuroSystemManager& m_nsManager;

			SIM_TRAIN_ENV m_train_env;
			SIM_RUN_ENV m_run_env;

			_layer_display_info_map m_layer_display_info_map;

			_uid_datanames_map m_last_learn_data_map;
			_uid_datanames_map m_last_test_data_map;

			_uid_datanames_map m_last_predict_data_map;
			std::wstring m_last_predict_output_path;
			dp::_STREAM_WRITE_INFO m_last_run_output_write_info;
		};
	}
}
