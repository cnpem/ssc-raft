#pragma once
#include <cuda.h>
#include <cufft.h>
#include <complex.h>
#include <tgmath.h> 
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <helper_cuda.h>
#include <time.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <sstream>
#include <string>
#include <cublas.h>
#include <future>
#include <chrono>
#include <thread>
#include <omp.h>
#include <math.h>
#include <cublas_v2.h>
#include "common/configs.h"
#include "common/structs.h"
#include "GeneralOperators/filter.h"
#include "GeneralOperators/alignment.h"
#include "geometries/gp/fbp.h"
#include "GeneralOperators/rings.h"
#include "GeneralOperators/flatdark.h"
#include "GeneralOperators/phasefilters.h"
#include "GeneralOperators/operators.h"
