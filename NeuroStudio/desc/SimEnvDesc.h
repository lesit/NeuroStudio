#pragma once

namespace np
{
	namespace str_rc
	{
		class SimEnvDesc
		{
		public:
			enum _DATA_SETUP_TYPE{
				DataSetup,
				TrainData, 
				RunData, 
				StartIndex, 
				EndIndex,
				UseTrainData_Run,
				NextTrainData_Run,
				StartIndex_Train_Desc,
				EndIndex_Train_Desc,
				StartIndex_Run_Desc,
				UseTrainData_Run_Desc,
				NextTrainData_Run_Desc
			};

			enum _TRAIN_SETUP_TYPE
			{
				TrainSetup,
				TrainRepeatCount,
				RepeatPerSeq,
				FullRepeat,
				LearnRate,
				RepeatPerSeq_Desc,
				FullRepeat_Desc,
				LearnRate_Desc,
			};

			enum _RUN_SETUP_TYPE
			{
				OutputStatistics,
				Tolerance,
				Tolerance_Desc,
			};
			static const wchar_t* GetDataSetupString(_DATA_SETUP_TYPE type);
			static const wchar_t* GetTrainSetupString(_TRAIN_SETUP_TYPE type);
			static const wchar_t* GetStatSetupString(_RUN_SETUP_TYPE type);
		};
	}
}