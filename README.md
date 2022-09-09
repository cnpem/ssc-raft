# sscRaft
----------------------------------------
Reconstruction Algorithms for Tomography
----------------------------------------

Authors:

	Eduardo X. Miqueles	Scientific Computing Group/LNLS/CNPEM
	
	Paola Ferraz Cunha	Scientific Computing Group/LNLS/CNPEM
	
	Gilberto Martinez Jr.	Scientific Computing Group/LNLS/CNPEM
	
	Giovanni Baraldi	Scientific Computing Group/LNLS/CNPEM

	Larissa M. Moreno	Mogno Beamline/LNLS/CNPEM

	Otávio M. Paiano	Mogno Beamline/LNLS/CNPEM
	 

## Install:

The prerequisite for installing sscRaft is `Python` and `CUDA` to run the routines.
The library sscRaft can be installed with either `pip` or `git`. 

### GIT

One can clone our [gitlab](https://gitlab.cnpem.br/) repository and install with the following steps

```bash
    git clone --recursive https://gitlab.cnpem.br/GCC/ssc-raft.git
    cd ssc-raft 
    python3 setup.py install --user --cuda
```

### PIP

One can install the latest version of sscRaft directly from our `pip server` 

```bash
pip config --user set global.extra-index-url http://gcc.lnls.br:3128/simple/
pip config --user set global.trusted-host gcc.lnls.br

pip install sscRaft
```

Or manually download it from the [package](https://gcc.lnls.br:3128/packages/) list


## Done:

1. `EM`: emission and transmission
2. `Rebinning`: conebeam to parallel rebinning
3. `FBP`: filter backprojection
4. `BST`: backprojection theorem
5. `Tomo360`: offset and 360 panoramic
6. `Rings`: rings removal
7. `Centersino`: parallel beam center alignment
8. `Filters`: filters


## To Do:
1. `FDK`: add cone beam filter backprojection
2. `WIGLLE`: add parallel beam projetion alignment
3. `Tomo360`: Correction of bug for odd angle dimension
4. `Aceleration`
5. `EM FST`
6. `EM`: bug for parallel beam 360 panoramic
