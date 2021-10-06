# ecc_python

This project provides python bindings for the computation of epipolar consistency in transmission imaging.
It is based on [pybind11](https://github.com/pybind/pybind11). This repo has been built upon their provided [cmake example](https://github.com/pybind/cmake_example).

The sources to be wrapped are located in [EpipolarConsistency](https://github.com/aaichert/EpipolarConsistency). Only the key functionality to compute the ECC quality metric 
from several projection images and their projection matrices is included in the python module. No optimizers, GUIs or any other advanced functionality from the basis repository is included. 

## Prerequisites

**On Unix (Linux, OS X)**

* A compiler with C++11 support
* CMake >= 3.4

**On Windows**

* Visual Studio 2015 or newer (required for all Python versions, see notes below)
* CMake >= 3.8 (3.8 was the first version to support VS 2015)


## Installation

Clone this repository (with `--recurse-submodules` option), install the necessary dependencies for the core ECC functionality in 
[EpipolarConsistency](https://github.com/aaichert/EpipolarConsistency) and pip install this repository. With the `setup.py` file 
included in this example, the `pip install` command will invoke CMake and build the pybind11 module as specified in `CMakeLists.txt`.

```bash
cd ecc_python
pip install .
```
It is crucial to have all dependencies of the core ECC functionality written in C++ installed properly. Please also refer to the 
Readme in [EpipolarConsistency](https://github.com/aaichert/EpipolarConsistency). On Ubuntu 20.04, a working set of dependencies
and their corresponding versions is:
- Eigen 3.3.9 (Even though Eigen is header only, in order to make sure that all files are located at the default locations, run
cmake and make install as described below. I know that Andre Aichert suggests to use Eigen 3.3.0, but I had issues with cmake not finding it then.)
- NLopt 2.6.2
- LibGetSet from Andre Aichert's fork 
- Qt 5.12.11 (Any version with major version number 5 should work. It is helpful to set the CMAKE_PREFIX_PATH to Qt as described [here](https://github.com/aaichert/EpipolarConsistency#41-notes-on-using-qt).)
- Cuda 11.2

The standard way to install most of these packages (like NLopt, GetSet) on Ubuntu is
```bash
mkdir build
cd build
cmake ..
make
sudo make install
```
For Windows, you need Visual Studio to build the solutions provided by cmake.


To test whether all dependencies are installed and found correctly, it might make sense to first build the submodule [EpipolarConsistency](https://github.com/aaichert/EpipolarConsistency)
separately and then try to pip install it as a python package.  

## Test call

```python
import ecc
help(ecc)
```

## Special notes for Windows

**Compiler requirements**

Pybind11 requires a C++11 compliant compiler, i.e Visual Studio 2015 on Windows.
This applies to all Python versions, including 2.7. Unlike regular C extension
modules, it's perfectly fine to compile a pybind11 module with a VS version newer
than the target Python's VS version. See the [FAQ] for more details.

**Runtime requirements**

The Visual C++ 2015 redistributable packages are a runtime requirement for this
project. It can be found [here][vs2015_runtime]. If you use the Anaconda Python
distribution, you can add `vs2015_runtime` as a platform-dependent runtime
requirement for you package: see the `conda.recipe/meta.yaml` file in this example.

## Some more comments

Only a small subset of functions in the EpipolarConsistency project is actually wrapped to python.

The functionality from EpipolarConsistency is compiled as a static library and linked into the shared python module.
For more details on this see https://github.com/pybind/cmake_example/issues/11#issuecomment-405092832.

## License

Pybind11 is provided under a BSD-style license that can be found in the LICENSE
file. By using, distributing, or contributing to this project, you agree to the
terms and conditions of this license.

