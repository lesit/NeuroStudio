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

#include <unordered_map>

#include "common.h"
#include "core/MathCoreApi.h"
#include "../LayerData.h"
#include "../../network/NeuralNetworkTypes.h"

namespace np
{
	using namespace network;
	namespace engine
	{
		namespace optimizer
		{
			/**
			 * base class of Optimizer
			 * usesHessian : true if an Optimizer uses hessian (2nd order derivative of loss function)
			 **/

			// constant value to avoid zero-division
			static const neuro_float eps = neuro_float(1e-8);
//			static const neuro_float eps = neuro_float(1e-20);
			class Optimizer
			{
			public:
				static Optimizer* CreateInstanceCPU(_optimizer_type type, const std::vector<neuro_float>& parameters);
				static Optimizer* CreateInstanceCUDA(_optimizer_type type, const std::vector<neuro_float>& parameters);

				Optimizer();
				virtual ~Optimizer() {}

				virtual _optimizer_type type() const = 0;

				virtual neuro_u32 GetHistoryCount() const { return 0; }
				virtual std::vector<neuro_float> GetParameters() const { return{}; }

				virtual bool Update(neuro_float learn_rate, const neuron_weight* dW, const _LAYER_INNER_DATA& weight) = 0;

				virtual void NextEpoch(){};
				virtual void reset() {} // override to implement pre-learning action

				virtual neuro_float default_learn_rate() const{ return 0.001f; }
			};

			class SGD : public Optimizer
			{
			public:
				SGD()
					: momentum(0)
				{
				}

				_optimizer_type type() const { return _optimizer_type::SGD; }
				neuro_u32 GetHistoryCount() const override { return 1; }

				neuro_float default_learn_rate() const override { return 0.01f; };

			protected:
				neuro_float momentum; // momentum
			};

			class Adagrad : public Optimizer
			{
			public:
				_optimizer_type type() const { return _optimizer_type::Adagrad; }
				neuro_u32 GetHistoryCount() const override { return 1; }

				neuro_float default_learn_rate() const override { return 0.01f; };
			};

			class Adam : public Optimizer
			{
			public:
				Adam(const std::vector<neuro_float>& parameters)
					: b1_t(neuro_float(0.9)), b2_t(neuro_float(0.999)), beta1(neuro_float(0.9)), beta2(neuro_float(0.999))
				{
					if (parameters.size() == 2)
					{
						b1_t = parameters[0];
						b2_t = parameters[1];
					}
				}

				_optimizer_type type() const { return _optimizer_type::Adam; }
				neuro_u32 GetHistoryCount() const override { return 2; }
				std::vector<neuro_float> GetParameters() const override { return{ b1_t, b2_t }; }

				neuro_float default_learn_rate() const override { return 0.001f; };

				void NextEpoch() override
				{
					b1_t *= beta1;
					b2_t *= beta2;
				}

				neuro_float b1_t; // decay term power t
				neuro_float b2_t; // decay term power t   

				neuro_float beta1; // decay term
				neuro_float beta2; // decay term
			};

			class RMSprop : public Optimizer
			{
			public:
				RMSprop()
					: decay(neuro_float(0.99))
				{
				}

				_optimizer_type type() const { return _optimizer_type::RMSprop; }
				neuro_u32 GetHistoryCount() const override { return 1; }

				neuro_float default_learn_rate() const override { return 0.0001f; };

				neuro_float decay; // decay term
			};

			static const neuro_float minimum_learn_rate = neuro_float(1e-10);
			static const neuro_float end_learn_rate = neuro_float(1e-20);
			class OptimizeInEpoch
			{
			public:
				static OptimizeInEpoch* CreateInstance(core::math_device_type pdtype, core::cuda::CudaInstance* cuda_instance
					, _optimizer_type type, const std::vector<neuro_float>& parameters, const network::_OPTIMIZING_RULE& rule);

				~OptimizeInEpoch();

				Optimizer* GetOptimizer(){
					return m_optimizer;
				}
				const network::_OPTIMIZING_RULE& GetRule() const { return m_rule; }

				bool Update(neuro_float lr_mult, neuron_weight* gradient, const _LAYER_INNER_DATA& weight);

				void NextBatch();
				void NextEpoch();
				neuro_float GetLearnRate() const{ return m_learn_rate; }

			protected:
				OptimizeInEpoch(core::math_device_type pdtype, core::cuda::CudaInstance* cuda_instance
					, Optimizer* optimizer, const network::_OPTIMIZING_RULE& _rule);

			private:
				core::MathCoreApi m_math;

				Optimizer* m_optimizer;
				network::_OPTIMIZING_RULE m_rule;

				neuro_float m_learn_rate; // learning rate

				neuro_size_t m_current_iterator;
			};
		}
	}
}
