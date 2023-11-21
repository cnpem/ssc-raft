#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages
from skbuild import setup

install_requires = [
    'cmake>=3.16',
    'scikit-build',
    'ninja'
]

# Main setup configuration.
setup(
    name='sscRaft',
    version=open('VERSION').read().strip(),
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    author=
    'Eduardo X. Miqueles / Paola Ferraz Cunha / Giovanni Baraldi / Gilberto Martinez Jr.',
    author_email='eduardo.miqueles@lnls.br',
    description='Reconstruction algorithms for tomography',
    keywords=['raft', 'tomography', 'radon', 'imaging'],
    url='http://www.',
    download_url='',
    license='BSD',
    platforms='Any',
    install_requires=install_requires,
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: LGPL-3.0 License',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education', 'Intended Audience :: Developers',
        'Natural Language :: English', 'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.0', 'Programming Language :: C',
        'Programming Language :: C++','Programming Language :: CUDA'
    ])
