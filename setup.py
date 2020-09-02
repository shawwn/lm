#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "absl-py==0.10.0",
    "pydantic==1.6.1",
    "jsonnet",
    "tensorflow>=1.15.2,<2.0",
    "mesh_tensorflow",
    "transformers==3.1.0",
    "tokenizers==0.8.1rc2",
]

setup_requirements = ["pytest-runner", "wheel"]

test_requirements = [
    "mypy",
    "pylint>=2.6.0",
    "pytest>=3",
]

setup(
    author="Fabrizio Milo",
    author_email="fmilo@entropysource.com",
    python_requires=">=3.6.9",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6.9",
        "Programming Language :: Python :: 3.7",
    ],
    description="End to End Language Model Pipeline built for training speed",
    entry_points={"console_scripts": ["lm=lm.cli.main:apprun",],},
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="lm, language models, pipeline, TPU, tensorflow",
    name="lm",
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/NeuroArchitect/lm",
    version="0.1.0",
    zip_safe=False,
)
