#include <stddef.h>

enum exeStatus {
    SUCCESS = 0,
    MALLOC_ERROR = -4, // CPU malloc error.
    INVALID_INPUT_AND_GPUS_CHECK_ERROR = -3, // got both errors below.
    GPUS_CHECK_ERROR = -2, // cudaError or a inapropriate device (e.g., not A100, H100 or better) was detected.
    INVALID_INPUT = -1,  // invalid user input.
    CUDA_CALL_ERROR = 1,  // sync error (non-sticky).
    CUDA_ASYNC_ERROR = 2, // async error (non-sticky).
    CUDA_GOT_CONTEXT_CORRUPTOR_ERROR = 3 // sticky error. may be sync or async.
};

// struct GPU {

// };