#ifndef RAFT_WORK_H
#define RAFT_WORK_H

#include "../common/structs.h"

extern "C"{

	WKP *allocateWorkspace(CFG configs, int blocksize);

	void freeWorkspace(WKP *workspace, CFG configs);

	Process *setProcesses(CFG configs, int i, int n_total_processes, int *gpus, int ngpus);

    void setProcessParallel(CFG configs, Process* process, int index, int n_total_processes, int *gpus, int ngpus);

    void setProcessConical(CFG configs, Process* process, int index, int n_total_processes, int *gpus, int ngpus);
    
    int getTotalProcesses(CFG configs, int ngpus);
}

#endif 