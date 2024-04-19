# SSC-RAFT: Reconstruction Algorithms for Tomography

## Authors

* Eduardo X. Miqueles, Scientific Computing Group/LNLS/CNPEM
* Paola Ferraz, Scientific Computing Group/LNLS/CNPEM
* Larissa M. Moreno, Scientific Computing Group/LNLS/CNPEM
* João F. G. de Albuquerque Oliveira, Scientific Computing Group/LNLS/CNPEM
* Alan Zanoni Peixinho, Scientific Computing Group/LNLS/CNPEM
* Yuri Rossi Tonin, Scientific Computing Group/LNLS/CNPEM
* Otávio M. Paiano, Scientific Computing Group/LNLS/CNPEM
* Gilberto Martinez Jr., former Scientific Computing Group/LNLS/CNPEM
* Giovanni Baraldi, former Scientific Computing Group/LNLS/CNPEM

### Contact

Sirius Scientific Computing Team: [gcc@lnls.br](malito:gcc@lnls.br)

## Documentation

`HTML` documentation can be found in the source directory `./docs/build/index.html` and can be opened with your preferred brownser.

## Install

This package uses `C`, `C++`, `CUDA` and `Python3`.
See bellow for full requirements.

The library `sscRaft` can be installed with form the source code or by `pip`/`git` if inside the CNPEM network.

## Source code from Zenodo

The source code can be downloaded from [zenodo website](https://zenodo.org/) under the DOI:[10.5281/zenodo.10988343](https://doi.org/10.5281/zenodo.10988343).

After download the `ssc-raft-v<version>.tar.gz` with the source files, one can decompress by

```bash
    tar -xvf ssc-raft-v<version>.tar.gz
```

To compile the source files, enter the follwing command inside the folder

```bash
    make clean && make
```

### GIT

One can clone our [gitlab](https://gitlab.cnpem.br/) repository and install with the following steps

```bash
    git clone --recursive https://gitlab.cnpem.br/GCC/ssc-raft.git --branch v<version> --single-branch
    cd ssc-raft 
    make clean && make
```

The `<version>` is the version of the `sscRaft` to be installed. Example, to install version 2.2.8

```bash
    git clone --recursive https://gitlab.cnpem.br/GCC/ssc-raft.git --branch v2.2.8 --single-branch
    cd ssc-raft 
    make clean && make
```

### PIP

If one is inside the CNPEM network, they can install the latest version of sscRaft directly from the `pip server`

```bash
    pip install sscRaft==version --index-url https://gitlab.cnpem.br/api/v4/projects/1978/packages/pypi/simple

```

Where `version` is the version number of the `sscRaft`

```bash
    pip install sscRaft==2.2.8 --index-url https://gitlab.cnpem.br/api/v4/projects/1978/packages/pypi/simple
```

More information on the `sscRaft` package on [sscRaft website](https://gcc.lnls.br/wiki/docs/ssc-raft/) and [sscRaft documentation](https://gcc.lnls.br/ssc/ssc-raft/index.html) available inside the CNPEM network.

## Memory

Be careful using GPU functions due to memory allocation.

## Requirements

Before installation, you will need the following packages installed:

* `CUDA >= 10.0.0`
* `C`
* `C++`
* `Python >= 3.8.0`
* `PIP`
* `libcurl4-openssl-dev`- Former Scientific Computing Group/LNLS/CNPEM

The following `SSC` modules are used:

* `ssc-commons`

The following modules are used:

* `CUBLAS`
* `CUFFT`
* `PTHREADS`
* `CMAKE>=3.18`

The following `Python3` modules are used:

* `scikit-build>=0.17.0`
* `setuptools>=64.0.0`
* `cython>=3.0.0`
* `numpy`
* `scikit-image >=0.19.3`
* `scipy`
* `matplotlib`
* `logging`
* `warning`
* `sys`
* `os`
* `pathlib`
* `inspect`
* `SharedArray`
* `ctypes`
* `uuid`
* `time`
* `h5py`
* `json`
* `multiprocessing`

## Uninstall

To uninstall `sscRaft` use the command, independent of the instalation method,

```bash
    pip uninstall sscRaft 
```
