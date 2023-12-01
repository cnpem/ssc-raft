#ifndef RAFT_WORK_H
#define RAFT_WORK_H

#include "../common/config.h"
#include "process.h"

typedef struct workspace
{	/* GPU */
	float *tomo, *recon;
	float *flat, *dark, *angles; 
}WKP;

extern "C"{

	WKP *allocateWorkspace(CFG configs, Process process);

	void freeWorkspace(WKP *workspace, CFG configs);

}

#endif 