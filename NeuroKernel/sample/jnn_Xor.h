#include "../NeuralNetworkProcessor/NeuralNetworkCreator.h"
#include "../NeuralNetworkProcessor/AHNeuralNetwork.h"

namespace ahnn
{
	namespace sample
	{
		class Xor
		{
		public:
			Xor(LPCTSTR strJNNFile)
				: m_strJNNFile(strJNNFile)
			{
			}

			void Create()
			{
				ahnn::network::NeuralNetworkCreator ahnnCreator(m_strJNNFile.c_str());

				ahnn::neuro_pointer32 iLayer=ahnnCreator.AddLayer();
				ahnnCreator.AddGroup(iLayer, 2);

				iLayer=ahnnCreator.AddLayer();
				ahnnCreator.AddGroup(iLayer, 100);

				iLayer=ahnnCreator.AddLayer();
				ahnnCreator.AddGroup(iLayer, 1);

				ahnnCreator.Connect(0, 0, 1, 0);
				ahnnCreator.Connect(1, 0, 2, 0);

				ahnnCreator.SaveNetwork();
			}

			void Train()
			{
				ahnn::network::AHNeuralNetwork ahnn(m_strJNNFile.c_str());

				ahnn::neuron_value inValue[8]={0,0,0,1,1,0,1,1};
				ahnn::neuron_value targetValue[4]={0,1,1,0};

				dataio::SimpleNeuroIOStream inData(inValue, 8, 2);
				dataio::SimpleNeuroIOStream targetData(targetValue, 4, 1);
				for(int iTrain=0;iTrain<10000;iTrain++)
				{
					if(!ahnn.Train(inData, targetData, false))
						break;

					if((iTrain+1)%100==0)
					{
						inData.MoveFirst();

						JNM_Util::DebugOutput(_T("%d times\r\n"), iTrain+1);
						//ahnn.DebugDisplay(inData, false);

						JNM_Util::DebugOutput(_T("\r\n"));
					}
					inData.MoveFirst();
					targetData.MoveFirst();
				}
			}

		private:
			std::wstring m_strJNNFile;
		};
	}
}
