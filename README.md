# SSC-RAFT: Reconstruction Algorithms for Tomography

## Authors

* Eduardo X. Miqueles, LNLS/CNPEM
* Paola Ferraz, LNLS/CNPEM

## Contributors

* Larissa M. Moreno, LNLS/CNPEM
* João F. G. de Albuquerque Oliveira, LNLS/CNPEM
* Alan Zanoni Peixinho, LNLS/CNPEM
* Yuri Rossi Tonin, LNLS/CNPEM

## Past contributors

* Otávio M. Paiano
* Gilberto Martinez Jr.
* Giovanni Baraldi
* Janito Vaqueiro Ferreira Filho
* Fernando S. Furusato
* Matheus F. Sarmento, LNLS/CNPEM
* Nikoley Koshev
* Elias Helou

## Acknowledgements

We would like to acknowledge the Brazilian Ministry of Science, Technology, and Innovation MCTI for supporting this work through the Brazilian Center for Research in Energy and Materials (CNPEM).
We want to thank Petrobras (Manhattan Project) - ID 2021/00018-3 for funding part of this project - and Mogno beamline from Sirius, for many fruitful discussion about the reconstruction strategies.
We also thank FAPESP/CEPID 2013/07375-0 for funding Nikolay Koshev, and FAPESP 2016/16238-4 for funding Gilberto Martinez Jr.

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

The `<version>` is the version of the `sscRaft` to be installed. Example, to install version 3.0.0

```bash
    git clone --recursive https://gitlab.cnpem.br/GCC/ssc-raft.git --branch v3.0.0 --single-branch
    cd ssc-raft 
    make clean && make
```

### PIP

If one is inside the CNPEM cluster, they can install the latest version of sscRaft directly from the `pip server`

```bash
    pip install sscRaft==version --index-url https://gitlab.cnpem.br/api/v4/projects/1978/packages/pypi/simple

```

Where `version` is the version number of the `sscRaft`

```bash
    pip install sscRaft==3.0.0 --index-url https://gitlab.cnpem.br/api/v4/projects/1978/packages/pypi/simple
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
* `libcurl4-openssl-dev`

This package supports nvidia ``GPUs`` with capabilities ``7.0`` or superior and a compiler with support to ``c++17``.

The following modules are used:

* `CUBLAS`
* `CUFFT`
* `PTHREADS`
* `CMAKE>=3.10`

The following `Python3` modules are used:

* `scikit-build>=0.17.0`
* `setuptools>=64.0.0`
* `numpy`
* `scikit-image >=0.19.3`
* `scipy`
* `matplotlib`
* `SharedArray`
* `h5py`

## Uninstall

To uninstall `sscRaft` use the command, independent of the instalation method,

```bash
    pip uninstall sscRaft 
```
