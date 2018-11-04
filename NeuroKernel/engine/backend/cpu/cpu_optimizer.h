/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY 
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND 
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include "../optimizer.h"

#include "../../../network/NeuralNetworkTypes.h"
#include "util/cpu_parallel_for.h"

namespace np
{
	namespace engine
	{
		namespace optimizer
		{
			namespace cpu
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
				class AdagradCpu : public Adagrad
				{
				public:
					bool Update(neuro_float learn_rate, const neuron_weight* dW, const _LAYER_INNER_DATA& weight) override;
				};

				/**	Adagrad의 단점을 보완
				 * RMSprop
				 *
				 * T Tieleman, and G E Hinton,
				 * Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine Learning (2012)
				 **/
				class RMSpropCpu : public RMSprop
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
				class AdamCpu : public Adam
				{
				public:
					AdamCpu(const std::vector<neuro_float>& parameters)
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
				class SGDCpu : public SGD
				{
				public:
					bool Update(neuro_float learn_rate, const neuron_weight* dW, const _LAYER_INNER_DATA& weight) override;
				};
			}
		}
	}
}

