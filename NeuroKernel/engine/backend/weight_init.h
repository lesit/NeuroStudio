/*
    Copyright (c) 2015, Taiga Nomi
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
#include "common.h"
#include "util/randoms.h"

namespace np
{
	namespace engine
	{
		namespace weight_init
		{
			class WeightInit
			{
			public:
				virtual void fill(neuro_size_t count, neuro_float* weight, neuro_size_t fan_in, neuro_size_t fan_out) const = 0;
			};

			class Scalable : public WeightInit {
			public:
				Scalable(neuro_float value) : scale(value) {}

				void Scale(neuro_float value) {
					scale = value;
				}
			protected:
				neuro_float scale;
			};

			/**
			 * Use fan-in and fan-out for scaling
			 *
			 * X Glorot, Y Bengio,
			 * Understanding the difficulty of training deep feedforward neural networks
			 * Proc. AISTATS 10, May 2010, vol.9, pp249-256
			 **/
			class Xavier : public Scalable {
			public:
				Xavier() : Scalable(neuro_float(6)) {}
				explicit Xavier(neuro_float value) : Scalable(value) {}

				void fill(neuro_size_t count, neuro_float* weight, neuro_size_t fan_in, neuro_size_t fan_out) const override
				{
					const neuro_float weight_base = std::sqrt(scale / (fan_in + fan_out));

					uniform_rand(weight, weight + count, -weight_base, weight_base);
				}
			};

			/**
			 * Use fan-in(number of input weight for each neuron) for scaling
			 *
			 * Y LeCun, L Bottou, G B Orr, and K Muller,
			 * Efficient backprop
			 * Neural Networks, Tricks of the Trade, Springer, 1998
			 **/
			class LeCun : public Scalable {
			public:
				LeCun() : Scalable(neuro_float(1)) {}
				explicit LeCun(neuro_float value) : Scalable(value) {}

				void fill(neuro_size_t count, neuro_float* weight, neuro_size_t fan_in, neuro_size_t fan_out) const override
				{
					NP_UNREFERENCED_PARAMETER(fan_out);

					const neuro_float weight_base = scale / std::sqrt(neuro_float(fan_in));

					uniform_rand(weight, weight + count, -weight_base, weight_base);
				}
			};

			class Gaussian : public Scalable {
			public:
				Gaussian() : Scalable(neuro_float(1)) {}
				explicit Gaussian(neuro_float sigma) : Scalable(sigma) {}

				void fill(neuro_size_t count, neuro_float* weight, neuro_size_t fan_in, neuro_size_t fan_out) const override
				{
					NP_UNREFERENCED_PARAMETER(fan_in);
					NP_UNREFERENCED_PARAMETER(fan_out);

					gaussian_rand(weight, weight + count, neuro_float(0), scale);
				}
			};

			class Constant : public Scalable {
			public:
				Constant() : Scalable(neuro_float(0)) {}
				explicit Constant(neuro_float value) : Scalable(value) {}

				void fill(neuro_size_t count, neuro_float* weight, neuro_size_t fan_in, neuro_size_t fan_out) const override
				{
					NP_UNREFERENCED_PARAMETER(fan_in);
					NP_UNREFERENCED_PARAMETER(fan_out);

					std::fill(weight, weight + count, scale);
				}
			};

			class He : public Scalable {
			public:
				He() : Scalable(neuro_float(2)) {}
				explicit He(neuro_float value) : Scalable(value) {}

				void fill(neuro_size_t count, neuro_float* weight, neuro_size_t fan_in, neuro_size_t fan_out) const override
				{
					NP_UNREFERENCED_PARAMETER(fan_out);

					const neuro_float sigma = std::sqrt(scale / fan_in);

					gaussian_rand(weight, weight + count, neuro_float(0), sigma);
				}
			};

		}
	}
}
