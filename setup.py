#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages
from skbuild import setup

from sscRaft import __version__

setup(
    packages=find_packages(),
    version=__version__,
    include_package_data=True,
    zip_safe=False,
    
    entry_points = {
        'console_scripts': [
            'ssc-raft = sscRaft.cli.ssc_raft_cli:app',
        ]
    }
    )

# from sscRaft import __version__

# install_requires = [
#     'ninja',
#     'click==8.0.4',
#     'colorama==0.4.5',
#     'rich==12.6.0',
#     'mdurl==0.1.0',
#     'Pygments==2.14.0',
#     'shellingham==1.5.0',
#     'typer==0.9.0',
#     'typing_extensions==4.1.1',
# ]

# # Main setup configuration.
# setup(
#     name = "sscRaft",
#     version = __version__,
#     packages=find_packages(),

#     include_package_data=True,

#     entry_points = {
#         'console_scripts': [
#             'ssc-raft = sscRaft.cli.ssc_raft_cli:app',
#         ]
#     },

#     zip_safe=False,
#     author='Eduardo X. Miqueles / Paola Ferraz / Larissa M. Moreno / Otávio Paiano / Alan Zanoni Peixinho / João Oliveira / Yuri Rossi Tonin / Giovanni Baraldi / Gilberto Martinez Jr.',
#     author_email='eduardo.miqueles@lnls.br, paola.ferraz@lnls.br',
#     description='Reconstruction algorithms for tomography',
#     keywords=['raft', 'tomography', 'radon', 'imaging', 'filtered backprojection', 'fdk', 'expectation-maximization','rings','tomogram alignment'],
#     url='https://gcc.lnls.br/ssc/ssc-raft/index.html',
#     download_url='',
#     platforms='Any',
#     install_requires=install_requires,
#     classifiers=[
#         'Development Status :: 4 - Beta',
#         'License :: OSI Approved :: LGPL-3.0 License',
#         'Intended Audience :: Science/Research',
#         'Intended Audience :: Education', 'Intended Audience :: Developers',
#         'Natural Language :: English', 'Operating System :: OS Independent',
#         'Programming Language :: Python',
#         'Programming Language :: Python :: 3.0', 'Programming Language :: C',
#         'Programming Language :: C++','Programming Language :: CUDA'
#     ])
