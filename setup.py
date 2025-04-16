import os
from setuptools import setup, find_packages

# Safer loading of requirements.txt
base_dir = os.path.abspath(os.path.dirname(__file__))
requirements_path = os.path.join(base_dir, "requirements.txt")

if os.path.exists(requirements_path):
    with open(requirements_path) as f:
        requirements = f.read().splitlines()
else:
    requirements = []

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="bakaano-hydro",
    version="1.1.2",
    author="Confidence Duku",
    author_email="confidence.duku@gmail.com",
    description="Distributed hydrology-guided neural network for streamflow prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/confidence-duku/bakaano-hydro",
    packages=find_packages(exclude=["tests*", "docs*"]),
    install_requires=requirements,
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Hydrology",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    keywords="flood forecasting hydrology deep learning distributed runoff tcn",
    license="Apache 2.0",
    project_urls={
        "Source": "https://github.com/confidence-duku/bakaano-hydro",
        "Bug Tracker": "https://github.com/confidence-duku/bakaano-hydro/issues",
        "Documentation": "https://github.com/confidence-duku/bakaano-hydro#readme",
    },
)
