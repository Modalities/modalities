from setuptools import find_packages, setup

with open("../README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='LLMgym',
    version='0.0.1',
    author='Foundation Model group  @ Fraunhofer IAIS',
    description="MLgym, a python framework for distributeda and reproducible machine learning model training in research.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        "pytest",
        "torch>=2.0",
        "tqdm",
        "pyyaml",
        "transformers",
        "datasets",
        "protobuf",
        "SentencePiece",
    ],
    python_requires=">=3.10",
    include_package_data=True,
)
