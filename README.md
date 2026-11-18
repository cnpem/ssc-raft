# SSC-RAFT: Reconstruction Algorithms for Tomography

<!-- about-start -->

The package `ssc-raft` is a tomography reconstruction framework developed by the Sirius Scientific Computing (GCC) Group at the Brazilian Center for Research in Energy and Materials (CNPEM).

**RAFT** (*Reconstruction Algorithms for Tomography*) has been under continuous development for more than a decade to support synchrotron imaging experiments at Sirius, the Brazilian fourth-generation synchrotron light source, as well as previous generation facilities. The package provides high-performance tools for tomographic image reconstruction, artifact correction, image processing, and large-scale data analysis. Implemented using a combination of low- and high-level programming languages, `ssc-raft` is designed to deliver fast, robust, and scalable reconstruction workflows for large imaging datasets.

Originally developed for the IMX beamline at the former UVX synchrotron, the software has evolved to address the computational challenges of modern synchrotron facilities. Today, it incorporates new software architectures and state-of-the-art reconstruction methods to support current and future Sirius beamlines, including Carnaúba, Cateretê, Mogno, Quati, and others.
Additional algorithms and research prototypes are continuously incorporated into the project as they mature and undergo validation by the GCC Group.

The source code is publicly archived on Zenodo under DOI [10.5281/zenodo.10988342](https://doi.org/10.5281/zenodo.10988342).

Example scripts are included to help users get started with the package.

> 💡 For questions, bug reports, or feature requests, contact the Sirius Scientific Computing Team at **gcc@lnls.br**.

## Authors and Contributors

### Authors

- Eduardo X. Miqueles (CNPEM, [ORCID](https://orcid.org/0000-0001-6576-4288))
- Paola Ferraz (CNPEM, [ORCID](https://orcid.org/0000-0002-1457-705X))

### Contributors

- João F. G. de Albuquerque Oliveira (CNPEM, [ORCID](https://orcid.org/0000-0003-3053-9608))
- Yuri Rossi Tonin ([ORCID](https://orcid.org/0000-0003-1403-815X))

### Past Contributors

- Larissa Macul Moreno ([ORCID](https://orcid.org/0000-0003-1627-093X))
- Alan Zanoni Peixinho ([ORCID](https://orcid.org/0009-0002-6983-2342))
- Otávio M. Paiano
- Gilberto Martinez Jr.
- Giovanni Baraldi
- Janito Vaqueiro Ferreira Filho
- Fernando S. Furusato
- Matheus F. Sarmento (CNPEM)
- Elias Helou
- Nikolay Koshev

## Acknowledgements

We would like to acknowledge the Brazilian Ministry of Science, Technology, and Innovation MCTI for supporting this work through the Brazilian Center for Research in Energy and Materials (CNPEM). 
We want to thank Petrobras (Manhattan Project) - ID 2021/00018-3 for funding part of this project - and Mogno beamline from Sirius, for many fruitful discussion about the reconstruction strategies. 
We also thank FAPESP/CEPID 2013/07375-0 for funding Nikolay Koshev, and FAPESP 2016/16238-4 for funding Gilberto Martinez Jr.

## Publications

The development of `ssc-raft` has resulted in the following publications:

1. Miqueles, E. X., Martinez Jr., G., and Guerrero, P. P. (2020). *Fast image reconstruction at a synchrotron laboratory*. Proceedings of the 2020 SIAM Conference on Parallel Processing for Scientific Computing, 24–34. DOI: [https://doi.org/10.1137/1.9781611976137.3](https://doi.org/10.1137/1.9781611976137.3)

2. Miqueles, E. X., Rinkel, J., O'Dowd, F., and Bermudez, J. S. V. (2014). *Generalized Titarenko's algorithm for ring artefacts reduction*. Journal of Synchrotron Radiation, 21, 1333–1346. DOI: [https://doi.org/10.1107/S1600577514016919](https://doi.org/10.1107/S1600577514016919)

3. Miqueles, E. X., Koshev, N., and Helou, E. S. (2018). *A Backprojection Slice Theorem for Tomographic Reconstruction*. IEEE Transactions on Image Processing, 27(2), 894–906. DOI: [https://doi.org/10.1109/TIP.2017.2766785](https://doi.org/10.1109/TIP.2017.2766785)

4. Miqueles, E. X. and Guerrero, P. P. (2020). *Automatic regularization for tomographic image reconstruction*. Results in Applied Mathematics, 6, 100088. DOI: [https://doi.org/10.1016/j.rinam.2019.100088](https://doi.org/10.1016/j.rinam.2019.100088)

5. Macul, L. and Miqueles, E. X. (2025). *Accelerating Tomographic Artifact Removal Using a Multi-GPU System*. In *CARLA 2024*, CCIS 2270. DOI: [https://doi.org/10.1007/978-3-031-80084-9_19](https://doi.org/10.1007/978-3-031-80084-9_19)

6. Ferraz, P., Paiano, O., and Miqueles, E. X. (2025). *Multi-GPU Tomographic Reconstructions of Large Volumes in the Frequency Domain*. In *CARLA 2024*, CCIS 2270. DOI: [https://doi.org/10.1007/978-3-031-80084-9_18](https://doi.org/10.1007/978-3-031-80084-9_18)

## Documentation

The HTML documentation is available in:

```text
docs/build/index.html
```

Open this file in your preferred web browser after building the documentation.

To build the documentation from source:

```shell
pip install -r docs/requirements.txt

cd docs
make
```

The generated documentation will be available in:

```text
docs/build/
```

<!-- about-end -->

## Installation

<!-- installation-start -->

The `ssc-raft` package can be installed from source code or directly via `pip` (within the CNPEM network).

It requires **C**, **C++**, **CUDA**, and **Python** to be installed before installation.
See the [Requirements](#requirements) section for a complete list of dependencies.

The `ssc-raft` package is imported as `sscRaft` in Python 

```python
import sscRaft
```

### Install from Pip

> ⚠️ **CNPEM Network Required**
>
> This installation method is available only within the CNPEM network.

> ❗ **Tepui Cluster Users**
>
> Before installing sscRaft, load the required modules: ` cuda/11.2.2`, `python/3.11.11` and `gcc/9.5.0-jld3zdv`.

Install the latest available version:

```shell
pip install sscRaft \
    --index-url https://gitlab.cnpem.br/api/v4/projects/1978/packages/pypi/simple
```

Install a specific version:

```shell
pip install sscRaft==3.0.3 \
    --index-url https://gitlab.cnpem.br/api/v4/projects/1978/packages/pypi/simple
```

Replace `3.0.3` with the desired version number.

### Install from GitLab

> ⚠️ **CNPEM Network Required**
>
> This installation method is available only within the CNPEM network.

> ❗ **Tepui Cluster Users**
>
> Before installing sscRaft, load the required modules: ` cuda/11.2.2`, `python/3.11.11` and `gcc/9.5.0-jld3zdv`.

Clone the repository and install the latest version:

```shell
git clone https://gitlab.cnpem.br/sirius/proc-sim-apps/sci/imaging/ssc-raft.git
cd ssc-raft
make clean && make
```

Install a specific version:

```shell
git clone https://gitlab.cnpem.br/sirius/proc-sim-apps/sci/imaging/ssc-raft.git \
    --branch v<version> --single-branch

cd ssc-raft
make clean && make
```

For example:

```shell
git clone https://gitlab.cnpem.br/sirius/proc-sim-apps/sci/imaging/ssc-raft.git \
    --branch v3.1.1 --single-branch

cd ssc-raft
make clean && make
```

### Install from GitHub

Clone the public GitHub repository and install the latest version:

```shell
git clone https://github.com/cnpem/ssc-raft.git
cd ssc-raft
make clean && make
```

Install a specific version:

```shell
git clone https://github.com/cnpem/ssc-raft.git \
    --branch v<version> --single-branch

cd ssc-raft
make clean && make
```

For example:

```shell
git clone https://github.com/cnpem/ssc-raft.git \
    --branch v3.1.1 --single-branch

cd ssc-raft
make clean && make
```

### Install from Zenodo Source Code

Source code releases are archived on Zenodo: [https://doi.org/10.5281/zenodo.10988342](https://doi.org/10.5281/zenodo.10988342)

Download the desired version of `ssc-raft.tar.gz` and extract it:

```shell
tar -xvf ssc-raft.tar.gz
```

Enter the extracted directory and build the package:

```shell
cd ssc-raft
make clean && make
```

## Requirements

Before installing sscRaft, ensure that the following dependencies are available:

| Dependency | Minimum Version |
|------------|----------------|
| CUDA | >= 11.0 |
| GCC / C++ Compiler | >= 9.0 |
| Python | >= 3.8 |
| CMake | >= 3.24 |
| pip | - |
| libcurl4-openssl-dev | - |
| cuBLAS | - |
| cuFFT | - |
| pthreads | - |

sscRaft supports NVIDIA GPUs with compute capability **7.0 or higher** and requires a compiler with **C++17** support.

### Python Dependencies

The following Python packages are required and installed automatically during installation:

- `scikit-build >= 0.17`
- `setuptools >= 61.0`
- `wheel >= 0.42.0`
- `numpy`
- `scikit-image >= 0.19.3`
- `scipy`
- `matplotlib`
- `SharedArray`
- `h5py`
- `matplotlib-scalebar`
- `cycler`
- `pyparsing`
- `python-dateutil`
- `click >= 8.0.4`
- `colorama >= 0.4.5`
- `rich >= 12.6.0`
- `mdurl >= 0.1.0`
- `Pygments >= 2.14.0`
- `shellingham >= 1.5.0`
- `typer >= 0.9.0`
- `typing_extensions >= 4.1.1`
- `tqdm`

## Uninstall

To uninstall sscRaft, regardless of the installation method used:

```shell
pip uninstall sscRaft
```

<!-- installation-end -->

## Troubleshooting

Build problems (for example, `cicc: not found` when compiling inside a `micromamba` environment in a Singularity/Apptainer container) are described in the documentation: `./docs/source/troubleshooting.rst`.
