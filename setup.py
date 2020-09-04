from setuptools import find_packages
from setuptools import setup


with open("README.md") as f:
    long_description = f.read()

setup(
    name="tf-spherical-hashing",
    version="0.0.1",
    description="Tensorflow implementation of Spherical Hashing.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="John Pertoft",
    author_email="john.pertoft@gmail.com",
    url="https://github.com/johnPertoft/spherical-hashing",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.0"
    ])
