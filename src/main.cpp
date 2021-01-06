#include <Eigen/Core>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "LibProjectiveGeometry/ProjectionMatrix.h"

namespace py = pybind11;

PYBIND11_MODULE(ecc, m)
{
    m.doc() = R"pbdoc(
        Python wrappers for epipolar consistency computations from https://github.com/aaichert/EpipolarConsistency
        -----------------------

        .. currentmodule:: ecc

        .. autosummary::
           :toctree: _generate

           makeCalibrationMatrix
    )pbdoc";

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

    //auto submodule_projective_geometry = m.def_submodule("projective_geometry")

    m.def("makeCalibrationMatrix", &Geometry::makeCalibrationMatrix, "Creates a new intrinsics matrix K.",
        py::arg("ax"), py::arg("ay"), py::arg("u0"), py::arg("v0"), py::arg("skew")=0.0);

}
