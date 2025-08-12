#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

with open("CLAUDE.md", "r") as f:
    long_description = f.read()

setup(
    name="featurewind",
    version="0.1.0",
    author="FeatureWind Team",
    description="A Python package for visualizing feature flows in high-dimensional data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    entry_points={
        "console_scripts": [
            "featurewind-basic=examples.basic_example:main",
        ],
    },
)