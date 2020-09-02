#!/usr/bin/env python
"""
Simple check list from AllenNLP repo: https://github.com/allenai/allennlp/blob/master/setup.py
To create the package for pypi.
1. Change the version in __init__.py, setup.py as well as docs/source/conf.py.
2. Unpin specific versions from setup.py (like isort).
2. Commit these changes with the message: "Release: VERSION"
3. Add a tag in git to mark the release: "git tag VERSION -m'Adds tag VERSION for pypi' "
   Push the tag to git: git push --tags origin master
4. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).
   For the wheel, run: "python setup.py bdist_wheel" in the top level directory.
   (this will build a wheel for the python version you use to build it).
   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions.
5. Check that everything looks correct by uploading the package to the pypi test server:
   twine upload dist/* -r pypitest
   (pypi suggest using twine as other methods upload files via plaintext.)
   You may have to specify the repository url, use the following command then:
   twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/
   Check that you can install it in a virtualenv by running:
   pip install -i https://testpypi.python.org/pypi transformers
6. Upload the final version to actual pypi:
   twine upload dist/* -r pypi
7. Copy the release notes from RELEASE.md to the tag in github once everything is looking hunky-dory.
8. Add the release version to docs/source/_static/js/custom.js and .circleci/deploy.sh
9. Update README.md to redirect to correct documentation.
"""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "absl-py==0.10.0",
    "pydantic==1.6.1",
    "jsonnet==0.16.0",
    "tensorflow>=1.15.2,<2.0",
    "mesh_tensorflow==0.1.16",
    "transformers@git+https://github.com/Mistobaan/transformers.git@3e8a0b2#egg=transformers-3.0.1",
    "tokenizers==0.8.1",
    "ftfy>=5.8,<5.9",
    "pyfarmhash==0.2.2"
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
    entry_points={"console_scripts": [
        "lm=lm.cli.main:apprun",
        "filter_lang=lm.cli.filter_lang:main",
        "train_tokenizer=lm.scripts.train_tokenizer:main"
        "convert_tokenizer=lm.scripts.convert_tokenizer:main"
        "check_dataset=lm.scripts.check_dataset:main"
        ],},
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
