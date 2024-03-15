"""Installation script for the 'shac' python package."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages

import os

root_dir = os.path.dirname(os.path.realpath(__file__))


# Minimum dependencies required prior to installation
INSTALL_REQUIRES = []

# Installation operation
setup(
    name="shac",
    author="Jie Xu",
    version="0.0.1",
    description="Short horizon actor critic",
    keywords=["robotics", "rl"],
    include_package_data=True,
    # python_requires=">=3.6.*", # commented out as it breaks things
    install_requires=INSTALL_REQUIRES,
    package_dir={"": "src"},
    packages=find_packages(
        where="src", exclude=["*.tests", "*.tests.*", "tests.*", "tests", "externals"]
    ),
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7, 3.8",
    ],
    zip_safe=False,
)

# EOF
