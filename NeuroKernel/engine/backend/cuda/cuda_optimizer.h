#if !defined(_CUDA_OPTIMIZER_H)
#define _CUDA_OPTIMIZER_H

#include "../optimizer.h"

#include "../../../network/NeuralNetworkTypes.h"
#include "util/cpu_parallel_for.h"

namespace np
{
	namespace engine
	{
		namespace optimizer
		{
			namespace cuda
			{
				/**
				* base class of Optimizer
				* usesHessian : true if an Optimizer uses hessian (2nd order derivative of loss function)
				**/

				// helper class to hold N values for each weight

				/**
				* adaptive gradient method
				*
				* J Duchi, E Hazan and Y Singer,
				* Adaptive subgradient methods for online learning and stochastic optimization
				* The Journal of Machine Learning Research, pages 2121-2159, 2011.
				**/
				class AdagradCuda : public Adagrad
				{
				public:
					bool Update(neuro_float learn_rate, const neuron_weight* dW, const _LAYER_INNER_DATA& weight) override;
				};

				/**
				* RMSprop
				*
				* T Tieleman, and G E Hinton,
				* Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine Learning (2012)
				**/
				class RMSpropCuda : public RMSprop
				{
				public:
					bool Update(neuro_float learn_rate, const neuron_weight* dW, const _LAYER_INNER_DATA& weight) override;
				};

				/**
				* @brief [a new Optimizer (2015)]
				* @details [see Adam: A Method for Stochastic Optimization (Algorithm 1)
				*               http://arxiv.org/abs/1412.6980]
				*
				*/
				class AdamCuda : public Adam
				{
				public:
					AdamCuda(const std::vector<neuro_float>& parameters)
						:Adam(parameters)
					{
					}

					bool Update(neuro_float learn_rate, const neuron_weight* dW, const _LAYER_INNER_DATA& weight) override;
				};

				/**
				* Stochast Gradient Descent
				*
				* B T Polyak,
				* Some methods of speeding up the convergence of iteration methods
				* USSR Computational Mathematics and Mathematical Physics, 4(5):1-17, 1964.
				* slightly faster when no momentum
				**/
				class SGDCuda : public SGD
				{
				public:
					bool Update(neuro_float learn_rate, const neuron_weight* dW, const _LAYER_INNER_DATA& weight) override;
				};
			}
		}
	}
}

#endif
