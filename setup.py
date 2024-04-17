#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages
from skbuild import setup

setup(
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    
    # entry_points = {
    #     'console_scripts': [
    #         'ssc-raft = sscRaft.cli.ssc_raft_cli:app',
    #     ]
    # }
    )
