# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import setuptools

setuptools.setup(
    name="dflex",
    version="0.0.1",
    author="NVIDIA",
    author_email="mmacklin@nvidia.com",
    description="Differentiable Multiphysics for Python",
    long_description="",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",  
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=["ninja", "torch", "numpy", "urchin", "usd-core"],
)
