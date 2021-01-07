#include <Eigen/Core>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h> //for automatic type conversion of Eigen to/from python
#include <pybind11/stl.h> // for automatic type conversion of STL containers to/from python

#include "LibProjectiveGeometry/ProjectionMatrix.h"
#include "LibEpipolarConsistency/EpipolarConsistencyRadonIntermediateCPU.hxx"

namespace py = pybind11;

PYBIND11_MODULE(ecc, m)
{
    m.doc() = "Python wrappers for epipolar consistency computations from https://github.com/aaichert/EpipolarConsistency";

    // contents of ProjectionMatrix.h
    m.def("makeCalibrationMatrix", &Geometry::makeCalibrationMatrix, "Creates a new intrinsics matrix K.",
        py::arg("ax"), py::arg("ay"), py::arg("u0"), py::arg("v0"), py::arg("skew")=0.0);
    m.def("makeProjectionMatrix", &Geometry::makeProjectionMatrix, "Creates a projection matrix from K, R and t.",
        py::arg("K"), py::arg("R"), py::arg("t"));
    m.def("computeFundamentalMatrix", &Geometry::computeFundamentalMatrix, "Computes fundamental matrix from two projection matrices.", 
        py::arg("P0"), py::arg("P1"));
    m.def("projectionMatrixDecomposition", &Geometry::projectionMatrixDecomposition, 
        "Decompose Projection Matrix into K[R|t] using RQ-decomposition. Returns false if R is left-handed (For RHS world coordinate systems, this implies imageVPointsUp is wrong).",
        py::arg("P"), py::arg("K"), py::arg("R"), py::arg("t"), py::arg("imageVPointsUp")=true);
    //TODO: why is python unhappy with normalizeProjectionMatrix??
    //m.def("normalizeProjectionMatrix", &Geometry::normalizeProjectionMatrix, 
    //    "Normalize a camera matrix P=[M|p4] by -sign(det(M))/||m3|| such that (du,dv,d)'=P(X,Y,Z,1)' encodes the depth d.",
    //    py::arg("P"));
    m.def("pseudoInverse", &Geometry::pseudoInverse, "Backprojection", py::arg("P"));
    m.def("getCameraIntrinsics", &Geometry::getCameraIntrinsics, "Return intrinsic parameters in upper triangular 3x3 matrix.",
        py::arg("P"));
    m.def("getCameraPrincipalPoint", &Geometry::getCameraPrincipalPoint, "Compute the principal point via M*m3.",
        py::arg("P"));
    m.def("getCameraDirection", &Geometry::getCameraDirection, "Direction of principal ray from a projection matrix. (Normal to principal plane, which is last row of P).",
        py::arg("P"));
    m.def("getCameraAxisDirections", &Geometry::getCameraAxisDirections, "Compute the two three-points where the image u- and v-axes meet infinity.",
        py::arg("P"));
    m.def("getCameraFocalLengthPx", &Geometry::getCameraFocalLengthPx, "Compute the focal length in pixels (diagonal entries of K in P=K[R t] ). Assumes normalized projection matrix.",
        py::arg("P"));
    m.def("getCameraImagePlane", &Geometry::getCameraImagePlane, "Decomposes the projection matrix to compute the equation of the image plane. For left-handed coordinate systems, pixel_spacing can be negated.",
        py::arg("P"), py::arg("pixel_spacing"));

    //TODO: How to convert from numpy to RadonIntermediates data structure?

    // EpipolarConsistencyRadonIntermediateCPU
    py::class_<EpipolarConsistency::MetricCPU>(m, "MetricCPU")
        .def(py::init<std::vector<Geometry::ProjectionMatrix>&, std::vector<EpipolarConsistency::RadonIntermediate*>, double>(), "Create MetricCPU instance.",
            py::arg("proj_mats"), py::arg("radon_derivatives"), py::arg("plane_angle_increment")=0.001745329251)
        .def("operator", py::overload_cast<>(&EpipolarConsistency::MetricCPU::operator()), "Evaluates consistency for all image pairs on CPU")
        .def("operator", py::overload_cast<int>(&EpipolarConsistency::MetricCPU::operator()), "Evaluates consistency for the i-th projection on CPU",
            py::arg("ref_projection"));

}
