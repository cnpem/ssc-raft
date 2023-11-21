# sscRaft
----------------------------------------
Reconstruction Algorithms for Tomography
----------------------------------------

Authors:

	Eduardo X. Miqueles	    Scientific Computing Group/LNLS/CNPEM
	
	Paola Cunha Ferraz 	    Scientific Computing Group/LNLS/CNPEM
	
	Gilberto Martinez Jr.	Scientific Computing Group/LNLS/CNPEM
	
	Giovanni Baraldi	    Scientific Computing Group/LNLS/CNPEM

	Larissa M. Moreno	    Scientific Computing Group/LNLS/CNPEM

	Otávio M. Paiano	    Scientific Computing Group/Mogno Beamline/LNLS/CNPEM

	Alan Z. Peixinho	    Scientific Computing Group/LNLS/CNPEM
	 

For more information on installation and usage we refer the user to the Scientific Computing Group (GCC) website [sscRaft Documentation](https://gcc.lnls.br/wiki/docs/ssc-raft/).

## Install:

The prerequisite for installing sscRaft is `Python` and `CUDA` to run the routines.
The library sscRaft can be installed with either `pip` or `git`. 

### GIT

One can clone our [gitlab](https://gitlab.cnpem.br/) repository and install with the following steps

```bash
    git clone https://gitlab.cnpem.br/GCC/ssc-raft.git --branch v<version> --single-branch
    cd ssc-raft 
    make clean & make
```

Example:

```bash
    git clone https://gitlab.cnpem.br/GCC/ssc-raft.git --branch v2.2.3 --single-branch
    cd ssc-raft 
    make clean & make
```

### PIP

One can install the latest version of sscRaft directly from our `pip server` 

```bash
	pip install sscRaft==<version> --index-url https://gitlab.cnpem.br/api/v4/projects/1978/packages/pypi/simple
```

Example:
```bash
	pip install sscRaft==2.2.3 --index-url https://gitlab.cnpem.br/api/v4/projects/1978/packages/pypi/simple
```






