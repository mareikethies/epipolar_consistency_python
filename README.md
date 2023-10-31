# Epipolar consistency for transmission imaging (Python bindings)

This repository provides python bindings for the computation of epipolar consistency conditions (ECC) as a metric in transmission imaging.
The ECC functionality itself is not implemented in this repository. 
The source files are located in [EpipolarConsistency](https://github.com/aaichert/EpipolarConsistency) and are written in C++ by André Aichert.
This repository only provides bindings to the existing source code such that it can be used in Python.
The bindings are based on [pybind11](https://github.com/pybind/pybind11) and this repository closely follows their provided [cmake example](https://github.com/pybind/cmake_example).

## Functionality
Only the core functionality of the original repository [EpipolarConsistency](https://github.com/aaichert/EpipolarConsistency) is included in the Python module. 
This includes 
- the initialization of images and projection matrices from numpy arrays,
- the computation of Radon intermediates,
- the evaluation of the ECC metric for all image pairs included in a given set of projection images and their corresponding projection matrices.

More advanced functionality of the original repository, that is *not* included in the bindings, includes
- everything related to optimization, 
- graphical user interfaces,
- everything related to 3D reconstruction.

We think that these aspects can similarly be handled by native Python libraries.  

## Example code

## Installation from wheel

This repository contains wheels for Linux operating system and Python 3.8 and 3.10. 
We strongly encourage to install the Python module directly from these wheels via, e.g.,  
```bash
pip install dist/ecc-0.0.1-cp310-cp310-linux_x86_64.whl
```

## Installation from source

If you need to compile the code on Windows or want to build it yourself for another reason, you need to 
(1) build the necessary dependencies for the core ECC functionality (see [EpipolarConsistency](https://github.com/aaichert/EpipolarConsistency))
(2) build and install the module in this repository.
On Linux, you need a compiler with C++11 support and CMake >= 3.5.
On Windows, Visual Studio 2015 or newer is required (Pybind11 requires a C++11 compliant compiler) as well as CMake >= 3.8 (3.8 was the first version to support VS 2015).

It is crucial to have all dependencies of the core ECC functionality written in C++ installed properly. Please also refer to the 
Readme in [EpipolarConsistency](https://github.com/aaichert/EpipolarConsistency). On Ubuntu 20.04, a working set of dependencies
and their corresponding versions is:
- Eigen 3.3.9 (Even though Eigen is header only, in order to make sure that all files are located at the default locations, run
cmake and make install as described below. I know that André Aichert suggests to use Eigen 3.3.0, but that did not work for me.)
- NLopt 2.6.2
- LibGetSet from André Aichert's fork 
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

After having installed the necessary dependencies for the core ECC functionality, you can finally install the Python module.
Clone this repository (with `--recurse-submodules` option) and run

```bash
cd ecc_python
pip install .
```

With the `setup.py` file included in this example, the `pip install` command will invoke CMake and build the pybind11 module as specified in `CMakeLists.txt`.
This will both build the module and install it to your current Python environment.

## Test call
To check if the Python module has been installed properly, you can run

```python
import ecc
help(ecc)
```

## Building a wheel

To build a wheel of the ecc package on Ubuntu, you need to have ninja installed (`sudo apt-get install ninja-build`) and then run
```bash
python setup.py bdist_wheel
```
Prebuild wheels for Linux with Python 3.8 and 3.10 are included in this repository. 

## Create html documentation of ecc module

To create an overview of all functions and classes provided by the ecc module, run

```bash
pydoc -w ecc
```
from an environment with ecc installed. It creates a file ecc.html within the same folder.

## Technical details

The functionality from EpipolarConsistency is compiled as a static library and linked into the shared python module.
For more details on this see https://github.com/pybind/cmake_example/issues/11#issuecomment-405092832.

To take full advantage of this code, an NVIDIA GPU is needed for a parallelized computation of Radon intermediates and the ECC metric itself.

## License

Pybind11 is provided under a BSD-style license that can be found in the LICENSE
file. By using, distributing, or contributing to this project, you agree to the
terms and conditions of this license.

## Citation

If you use this code for your research, please cite the original work on ECC in transmission imaging by André Aichert:
```
@ARTICLE{7094279,
  author={Aichert, André and Berger, Martin and Wang, Jian and Maass, Nicole and Doerfler, Arnd and Hornegger, Joachim and Maier, Andreas K.},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Epipolar Consistency in Transmission Imaging}, 
  year={2015},
  volume={34},
  number={11},
  pages={2205-2219},
  doi={10.1109/TMI.2015.2426417}}
```

