#ifndef REB_H
#define REB_H

#include "common/structs.h"
    
extern "C"{
    void GPUrebinning(float *conetomo, float *tomo, float *parameters, size_t *volumesize, int *gpus, int ngpus);
}

extern "C"{
    void CPUrebinning(float *conetomo, float *tomo, float *parameters, size_t *volumesize);
}

extern "C"{
    void cone_rebinning(RDAT *reb, PAR param);
}

extern "C"{
    void _GPUrebinning(PAR param, PROF *prof, float *conetomo, float *tomo, int ngpus);
}
extern "C"{
    void set_rebinning_parameters_gpu(PAR *param, float *parameters, size_t *volumesize, int *gpus);
}

extern "C"{
    void set_rebinning_parameters_cpu(PAR *param, float *parameters, size_t *volumesize);
}

extern "C"{
    RDAT *allocate_gpu_rebinning(RDAT *workspace, PAR param);
}

extern "C"{
    void free_gpu_rebinning(RDAT *reb);
}

extern "C"{
    void cone_rebinning_cpu(float *tomo, float *conetomo, PAR param);
}

extern "C"{
    __global__ void cone_rebinning_kernel(float *dtomo, float *dctomo, PAR param);
}

#endif 