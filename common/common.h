#if !defined(_NP_COMMON_H)
#define _NP_COMMON_H

#include <stdarg.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include <algorithm>
#include <minmax.h>

#include "platform_env.h"
#include "np_types.h"
#include "data_vector.h"
#include "tensor/tensor_shape.h"
#include "tensor/tensor_data.h"
#include "util/np_util.h"
#include "util/StringUtil.h"

#define BYTES_PER_KB	1024LL
#define BYTES_PER_MB	1048576LL
#define BYTES_PER_GB	1073741824LL
#define BYTES_PER_TB	1099511627776LL
#define BYTES_PER_PB	1125899906842624LL

#define MAX_NEURON			4225000000LL
#define MAX_NEURON_INPUT	65000

#define NP_UNREFERENCED_PARAMETER(x) (void)(x)

using namespace np;
using namespace np::tensor;

#ifdef _DEBUG
	#ifndef _CRTDBG_MAP_ALLOC
	#define _CRTDBG_MAP_ALLOC #include <stdlib.h> #include <crtdbg.h>  
	#endif
#endif  // _DEBUG  

#endif
