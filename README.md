# sscRaft
----------------------------------------
Reconstruction Algorithms for Tomography
----------------------------------------

Authors:

	Eduardo X. Miqueles	Scientific Computing Group/LNLS/CNPEM
	
	Paola Ferraz Cunha	Scientific Computing Group/LNLS/CNPEM
	
	Gilberto Martinez Jr.	Scientific Computing Group/LNLS/CNPEM
	
	Giovanni Baraldi	Scientific Computing Group/LNLS/CNPEM

	Larissa M. Moreno	Scientific Computing Group/LNLS/CNPEM

	Otávio M. Paiano	Mogno Beamline/LNLS/CNPEM
	

More information on the `sscRaft` package on [sscRaft website](https://gcc.lnls.br/wiki/docs/ssc-raft/).

## Install:

This package uses `C`, `C++`, `CUDA`, `CUBLAS`, `CUFFT`, `PTHREADS` 
and `PYTHON3`. The requirements for installing `sscRaft` are found in the `requirement.txt` file.

The library sscRaft can be installed with either `pip` or `git`. 


### GIT

One can clone our [gitlab](https://gitlab.cnpem.br/) repository and install with the following steps

```bash
    git clone https://gitlab.cnpem.br/GCC/ssc-raft.git --branch v<version> --single-branch
    cd ssc-raft 
    python3 -m pip install -r requirements.txt
    make clean && make
```

The `<version>` is the version of the `sscRaft` to be installed. Example, to install version 3.0.0

```bash
    git clone https://gitlab.cnpem.br/GCC/ssc-raft.git --branch v3.0.0 --single-branch
    cd ssc-raft 
    python3 -m pip install -r requirements.txt
    make clean && make
```


### PIP

One can install the latest version of sscRaft directly from our `pip server` 

```bash
pip install sscRaft==version --index-url https://gitlab.cnpem.br/api/v4/projects/1978/packages/pypi/simple

```

Where `version` is the version number of the `sscRaft`

```bash
pip install sscRaft==3.0.0 --index-url https://gitlab.cnpem.br/api/v4/projects/1978/packages/pypi/simple
```

## MEMORY

Be careful using GPU functions due to memory allocation.

## UNINSTALL

To uninstall `sscRaft` use the command

```bash
    pip uninstall sscRaft -y
```